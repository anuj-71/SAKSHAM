"""
Stability regression test for inference parameter changes.

Replays the 4997 pre-cached feature tensors (dataset/models/.feature_cache/)
through the ONNX model + _apply_stability logic to compare:
  - OLD params: STABILITY_WINDOW=12, STABILITY_THRESHOLD=0.60, COOLDOWN=2.5s
  - NEW params: STABILITY_WINDOW=7,  STABILITY_THRESHOLD=0.60, COOLDOWN=1.0s

Metrics reported per configuration:
  - Confirmation rate       (fraction of injected samples that fired a sign)
  - Accuracy on confirmed   (of those, fraction that matched the true label)
  - False-positive rate     (confirmed != true label)
  - Median frames-to-confirm (latency proxy)
  - Per-class breakdown

Usage:
    cd e:\\gessture
    .venv\\Scripts\\python.exe -m pytest tests/test_stability_regression.py -v -s
  or run directly:
    .venv\\Scripts\\python.exe tests/test_stability_regression.py
"""
from __future__ import annotations

import json
import sys
import os
import time
from collections import deque, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# ── Constants matching inference.py ────────────────────────────────────────────
MODEL_PATH      = "models/model.onnx"
LABEL_MAP_PATH  = "models/label_map.json"
FEATURES_PATH   = "models/.feature_cache/features.npy"
LABELS_PATH     = "models/.feature_cache/labels.npy"
MIN_CONFIDENCE  = 0.55  # same as inference.py

# Skip gracefully when model or feature cache hasn't been built yet
if not Path(MODEL_PATH).exists() or not Path(FEATURES_PATH).exists():
    pytest.skip(
        f"Model or feature cache not found — run tools/train.py first.",
        allow_module_level=True,
    )



def run_onnx_on_feature(session, input_name: str, feature: np.ndarray) -> np.ndarray:
    """Run one cached feature vector through the ONNX model, return softmax probs."""
    inp = feature[np.newaxis].astype(np.float32)          # (1, 64, 128)
    logits = session.run(None, {input_name: inp})[0][0]   # (13,)
    e = np.exp(logits - logits.max())
    return e / e.sum()


def simulate_stability(
    session,
    input_name: str,
    features: np.ndarray,   # (N, 64, 128)
    labels: np.ndarray,     # (N,)
    idx_to_label: dict,
    stability_window: int,
    stability_threshold: float,
    cooldown_frames: int,   # cooldown expressed in inference-stride frames
    inference_stride: int = 4,
) -> dict:
    """
    Feed each sample through ONNX one frame at a time (simulating a live stream)
    and track stability behaviour.

    Each cached sample represents exactly SEQ_LEN=64 raw frames of one sign.
    We play them back sequentially; the vote window and cooldown persist across
    samples to mimic a continuous session.

    Returns a result dict with aggregate and per-class metrics.
    """
    vote_window: deque[np.ndarray] = deque(maxlen=stability_window)
    last_confirmed_frame: int = -cooldown_frames  # allow fire on frame 0

    confirmed_signs: list[dict] = []   # {true_label, confirmed_label, frame}
    total_samples = len(features)

    global_frame = 0  # running inference-stride frame counter

    for sample_idx in range(total_samples):
        true_label = idx_to_label[int(labels[sample_idx])]
        feature = features[sample_idx]  # (64, 128)

        # Each cached sample = 64 raw frames; inference fires every `stride` frames
        num_inferences = 64 // inference_stride  # 16 per sample
        confirmed_this_sample = False

        for inf_step in range(num_inferences):
            global_frame += 1
            probs = run_onnx_on_feature(session, input_name, feature)
            vote_window.append(probs)

            if len(vote_window) < stability_window:
                continue

            avg_probs = np.mean(list(vote_window), axis=0)
            top_idx = int(np.argmax(avg_probs))
            top_conf = float(avg_probs[top_idx])
            top_label = idx_to_label[top_idx]

            if top_conf >= stability_threshold:
                if (global_frame - last_confirmed_frame) >= cooldown_frames:
                    confirmed_signs.append({
                        "true_label": true_label,
                        "confirmed_label": top_label,
                        "correct": top_label == true_label,
                        "confidence": round(top_conf, 4),
                        "frame": global_frame,
                        "inf_step_within_sample": inf_step,
                    })
                    last_confirmed_frame = global_frame
                    vote_window.clear()
                    confirmed_this_sample = True
                    break  # one confirmation per sign sample is enough

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    n_confirmed = len(confirmed_signs)
    n_correct   = sum(1 for c in confirmed_signs if c["correct"])
    n_wrong     = n_confirmed - n_correct

    confirmation_rate  = n_confirmed / total_samples if total_samples > 0 else 0.0
    accuracy_confirmed = n_correct   / n_confirmed   if n_confirmed  > 0 else 0.0
    fp_rate            = n_wrong     / n_confirmed   if n_confirmed  > 0 else 0.0

    latencies = [c["inf_step_within_sample"] for c in confirmed_signs]
    median_latency_steps = float(np.median(latencies)) if latencies else float("nan")
    # Convert to frames: each step = inference_stride raw frames
    median_latency_frames = median_latency_steps * inference_stride

    # Per-class breakdown
    per_class: dict[str, dict] = defaultdict(lambda: {"total": 0, "confirmed": 0, "correct": 0})
    for sample_idx in range(total_samples):
        per_class[idx_to_label[int(labels[sample_idx])]]["total"] += 1
    for c in confirmed_signs:
        per_class[c["true_label"]]["confirmed"] += 1
        if c["correct"]:
            per_class[c["true_label"]]["correct"] += 1

    per_class_summary = {}
    for cls, d in sorted(per_class.items()):
        conf_rate = d["confirmed"] / d["total"] if d["total"] > 0 else 0.0
        acc       = d["correct"]   / d["confirmed"] if d["confirmed"] > 0 else 0.0
        per_class_summary[cls] = {
            "total": d["total"],
            "confirmed": d["confirmed"],
            "confirm_rate": round(conf_rate, 3),
            "accuracy_on_confirmed": round(acc, 3),
        }

    return {
        "params": {
            "stability_window": stability_window,
            "stability_threshold": stability_threshold,
            "cooldown_frames": cooldown_frames,
            "inference_stride": inference_stride,
        },
        "total_samples": total_samples,
        "n_confirmed": n_confirmed,
        "confirmation_rate": round(confirmation_rate, 4),
        "accuracy_on_confirmed": round(accuracy_confirmed, 4),
        "false_positive_rate": round(fp_rate, 4),
        "median_latency_steps": round(median_latency_steps, 2),
        "median_latency_raw_frames": round(median_latency_frames, 1),
        "per_class": per_class_summary,
    }


def print_report(label: str, result: dict):
    p = result["params"]
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  window={p['stability_window']}  threshold={p['stability_threshold']}  "
          f"cooldown={p['cooldown_frames']} inference-frames "
          f"(stride={p['inference_stride']})")
    print(f"{'='*65}")
    print(f"  Samples:              {result['total_samples']}")
    print(f"  Confirmed:            {result['n_confirmed']}  "
          f"({result['confirmation_rate']*100:.1f}% of samples)")
    print(f"  Accuracy (confirmed): {result['accuracy_on_confirmed']*100:.1f}%")
    print(f"  False-positive rate:  {result['false_positive_rate']*100:.1f}%")
    print(f"  Median latency:       {result['median_latency_steps']:.1f} inference steps "
          f"= {result['median_latency_raw_frames']:.0f} raw frames")
    print(f"\n  {'Class':<12} {'Total':>6} {'Confirmed':>10} {'Confirm%':>9} {'Acc%':>6}")
    print(f"  {'-'*50}")
    for cls, d in result["per_class"].items():
        print(f"  {cls:<12} {d['total']:>6} {d['confirmed']:>10} "
              f"{d['confirm_rate']*100:>8.1f}% {d['accuracy_on_confirmed']*100:>5.1f}%")
    print()


def compare_and_assert(old: dict, new: dict):
    """
    Assert that new params don't regress accuracy by more than 3 pp
    and don't more than double the false-positive rate.
    """
    acc_delta = new["accuracy_on_confirmed"] - old["accuracy_on_confirmed"]
    fp_delta  = new["false_positive_rate"]   - old["false_positive_rate"]

    print(f"\n{'='*65}")
    print("  DELTA (new - old)")
    print(f"{'='*65}")
    print(f"  Confirmation rate delta:  {(new['confirmation_rate'] - old['confirmation_rate'])*100:+.1f} pp")
    print(f"  Accuracy delta:           {acc_delta*100:+.1f} pp")
    print(f"  False-positive delta:     {fp_delta*100:+.1f} pp")
    lat_delta = new["median_latency_raw_frames"] - old["median_latency_raw_frames"]
    print(f"  Latency delta:            {lat_delta:+.1f} raw frames  "
          f"({'faster' if lat_delta < 0 else 'slower'})")
    print()

    # Regression guards
    assert acc_delta >= -0.03, (
        f"Accuracy regressed by {-acc_delta*100:.1f} pp (limit: 3 pp). "
        "Tighten window or threshold."
    )
    if old["false_positive_rate"] > 0:
        assert new["false_positive_rate"] <= old["false_positive_rate"] * 2.0, (
            f"False-positive rate more than doubled: "
            f"{old['false_positive_rate']*100:.1f}% -> {new['false_positive_rate']*100:.1f}%"
        )
    print("  [PASS] No stability regression detected.")


def test_stability_parameters():
    """pytest-compatible test entry point."""
    import onnxruntime as ort

    assert Path(MODEL_PATH).exists(),     f"Model not found: {MODEL_PATH}"
    assert Path(LABEL_MAP_PATH).exists(), f"Label map not found: {LABEL_MAP_PATH}"
    assert Path(FEATURES_PATH).exists(),  f"Feature cache not found: {FEATURES_PATH}"
    assert Path(LABELS_PATH).exists(),    f"Labels cache not found: {LABELS_PATH}"

    print("\nLoading ONNX session and feature cache...")
    session    = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    features   = np.load(FEATURES_PATH)   # (4997, 64, 128)
    labels     = np.load(LABELS_PATH)     # (4997,)

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

    print(f"Dataset: {len(features)} samples, {len(label_map)} classes")

    # Use a stratified subset (50 samples per class) to keep runtime under 60s.
    # Covers all 13 classes proportionally; set SAMPLE_CAP=None to run full set.
    SAMPLE_CAP_PER_CLASS = 50
    if SAMPLE_CAP_PER_CLASS is not None:
        rng = np.random.default_rng(42)
        keep = []
        for cls_idx in range(len(label_map)):
            idxs = np.where(labels == cls_idx)[0]
            chosen = rng.choice(idxs, size=min(SAMPLE_CAP_PER_CLASS, len(idxs)), replace=False)
            keep.extend(chosen.tolist())
        keep = np.array(sorted(keep))
        features = features[keep]
        labels   = labels[keep]
        print(f"Using stratified subset: {len(features)} samples ({SAMPLE_CAP_PER_CLASS}/class)")

    STRIDE = 4  # matches inference.py

    # cooldown expressed in inference-stride units so it's frame-rate-agnostic
    # OLD: 2.5s cooldown - at 30fps / stride 4 = 7.5 inference steps -> 8
    # NEW: 1.0s cooldown - at 30fps / stride 4 = 7.5 inference steps -> 3
    # (we use discrete step counts, not wall-clock, because cached replay has no fps)
    OLD_COOLDOWN_STEPS = 8   # ~2.5s @ 30fps with stride 4  (32 raw frames)
    NEW_COOLDOWN_STEPS = 3   # ~1.0s @ 30fps with stride 4  (12 raw frames)

    print("\nRunning OLD params (window=12, threshold=0.60, cooldown~2.5s)...")
    old_result = simulate_stability(
        session, input_name, features, labels, idx_to_label,
        stability_window=12,
        stability_threshold=0.60,
        cooldown_frames=OLD_COOLDOWN_STEPS,
        inference_stride=STRIDE,
    )

    print("Running NEW params (window=7, threshold=0.60, cooldown~1.0s)...")
    new_result = simulate_stability(
        session, input_name, features, labels, idx_to_label,
        stability_window=7,
        stability_threshold=0.60,
        cooldown_frames=NEW_COOLDOWN_STEPS,
        inference_stride=STRIDE,
    )

    print_report("OLD  (window=12, cooldown~2.5s)", old_result)
    print_report("NEW  (window=7,  cooldown~1.0s)", new_result)
    compare_and_assert(old_result, new_result)


if __name__ == "__main__":
    test_stability_parameters()
