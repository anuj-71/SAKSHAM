"""
Batch MediaPipe Hands landmark extraction for CISLR.

Outputs per-video NPZ files in canonical SAKSHAM format:
  left_hand       (T, 21, 3)  float32  — raw MediaPipe landmarks
  right_hand      (T, 21, 3)  float32  — raw MediaPipe landmarks
  left_present    (T,)        float32  — 1.0 per frame if hand detected
  right_present   (T,)        float32  — 1.0 per frame if hand detected
  left_confidence (T,)        float32  — detection confidence
  right_confidence(T,)        float32  — detection confidence
  timestamps      (T,)        float32  — sequential frame index
  gloss           str                  — normalized gloss label
  signer_id       str                  — from CSV metadata
  uid             str                  — video UID
  video_path      str                  — resolved filesystem path
  frame_count     int                  — total frames in source video

Usage:
    .\.venv\Scripts\python.exe -m tools.cislr.extract_landmarks --limit 0

"""
import argparse
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from tools.cislr.utils import (
    ensure_dirs,
    DATASET_CSV,
    LABEL_FREQ_CSV,
    LANDMARKS_DIR,
    get_video_resolver,
    normalize_label,
)

from src.sign_language.tracker import HandTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def write_label_freq(counter: Counter, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gloss", "count"])
        for gloss, cnt in counter.most_common():
            writer.writerow([gloss, cnt])


def extract_from_video(
    video_path: str, tracker: HandTracker
) -> Tuple:
    """Extract raw MediaPipe landmarks from every frame of a video.

    Returns (left_hand, right_hand, left_present, right_present,
             left_confidence, right_confidence, timestamps, opened, frame_count).

    All frames are recorded — absent hands produce zero landmarks + 0.0 presence.
    Landmarks are raw (not normalized, not flattened).
    """
    cap = cv2.VideoCapture(video_path)
    left_hand_list: List[np.ndarray] = []
    right_hand_list: List[np.ndarray] = []
    left_present_list: List[float] = []
    right_present_list: List[float] = []
    left_conf_list: List[float] = []
    right_conf_list: List[float] = []
    missing_hand = np.zeros((21, 3), dtype=np.float32)

    if not cap.isOpened():
        logging.warning("Unable to open video: %s", video_path)
        return (None, None, None, None, None, None, None, False, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected, data = tracker.process_frame(frame)

        l_land = missing_hand
        r_land = missing_hand
        l_pres = 0.0
        r_pres = 0.0
        l_conf = 0.0
        r_conf = 0.0

        if detected:
            left = data.get("left_hand", {})
            right = data.get("right_hand", {})

            if left.get("present"):
                lm = left["landmarks"]
                if len(lm) >= 21:
                    l_land = np.array(lm[:21], dtype=np.float32)
                    l_pres = 1.0
                    l_conf = float(left.get("confidence", 0.0))

            if right.get("present"):
                lm = right["landmarks"]
                if len(lm) >= 21:
                    r_land = np.array(lm[:21], dtype=np.float32)
                    r_pres = 1.0
                    r_conf = float(right.get("confidence", 0.0))

        left_hand_list.append(l_land)
        right_hand_list.append(r_land)
        left_present_list.append(l_pres)
        right_present_list.append(r_pres)
        left_conf_list.append(l_conf)
        right_conf_list.append(r_conf)

    cap.release()

    frame_count = len(left_hand_list)
    if frame_count == 0:
        return (None, None, None, None, None, None, None, True, 0)

    return (
        np.stack(left_hand_list, axis=0).astype(np.float32),
        np.stack(right_hand_list, axis=0).astype(np.float32),
        np.array(left_present_list, dtype=np.float32),
        np.array(right_present_list, dtype=np.float32),
        np.array(left_conf_list, dtype=np.float32),
        np.array(right_conf_list, dtype=np.float32),
        np.arange(frame_count, dtype=np.float32),
        True,
        frame_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process (0 = all)")
    parser.add_argument("--vocab-size", type=int, default=40, help="Recommended V1 vocab size")
    args = parser.parse_args()

    ensure_dirs()

    # read dataset CSV
    rows = []
    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total_videos = len(rows)
    logging.info("Dataset rows: %d", total_videos)

    # build gloss frequency
    freq = Counter()
    per_gloss_rows = defaultdict(list)
    for r in rows:
        uid = r.get("uid") or r.get("id") or list(r.values())[0]
        gloss = normalize_label(r.get("gloss", r.get("label", "")))
        freq[gloss] += 1
        per_gloss_rows[gloss].append((uid, r))

    write_label_freq(freq, LABEL_FREQ_CSV)
    logging.info("Wrote label frequency to %s", LABEL_FREQ_CSV)

    # recommend V1 vocab
    recommended = [g for g, _ in freq.most_common(args.vocab_size)]

    # choose HELLO replacement if HELLO not present or low-count
    hello_present = "HELLO" in freq
    hello_replacement = None
    if not hello_present or freq.get("HELLO", 0) < 2:
        # candidate greetings
        keywords = ("SEE", "YOU", "TOMORROW", "GOOD", "MORNING", "BYE", "HI", "WELCOME", "GREETING")
        candidates = [g for g in freq.keys() if any(k in g for k in keywords) and g != "HELLO"]
        if candidates:
            # pick most frequent
            candidates.sort(key=lambda g: freq[g], reverse=True)
            hello_replacement = candidates[0]
            logging.info("Selected HELLO replacement: %s", hello_replacement)

    # prepare tracker and resolver
    video_root, resolver = get_video_resolver()
    if resolver is None:
        logging.error("Video resolver could not locate CISLR video root. Aborting.")
        return 2

    # Initialize HandTracker with a clear error message if MediaPipe API is incompatible
    try:
        tracker = HandTracker()
    except Exception as e:
        logging.error("Failed to initialize HandTracker: %s", e)
        logging.error("This usually indicates an incompatible mediapipe installation (missing mp.solutions).")
        logging.error("Check mediapipe version and installation. Example checks (use the project venv):\n  .\\.venv\\Scripts\\python.exe -c \"import mediapipe as mp; print(mp.__version__, mp.__file__, hasattr(mp, 'solutions'))\"")
        return 3

    # output dirs
    out_dir = Path(LANDMARKS_DIR)
    accepted_dir = out_dir / "accepted"
    rejected_dir = out_dir / "rejected"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "processed": 0,
        "opened": 0,
        "success": 0,
        "failed": 0,
    }
    per_gloss_counts = Counter()

    limit = args.limit or 0
    for idx, r in enumerate(rows):
        if limit and stats["processed"] >= limit:
            break

        uid = r.get("uid") or r.get("id") or list(r.values())[0]
        gloss = normalize_label(r.get("gloss", r.get("label", "")))

        video_path = resolver.resolve(f"{uid}.mp4")
        stats["processed"] += 1

        result = extract_from_video(video_path, tracker)
        left_hand, right_hand, left_present, right_present, left_conf, right_conf, timestamps, opened, frame_count = result
        if not opened:
            stats["failed"] += 1
            (rejected_dir / f"{uid}.npz").write_bytes(b"")
            continue

        # Reject if no hands present in any frame
        if np.max(left_present) == 0.0 and np.max(right_present) == 0.0:
            stats["failed"] += 1
            (rejected_dir / f"{gloss}_{uid}.npz").write_bytes(b"")
            continue

        # signer metadata if available in row
        signer_id = r.get("signer") or r.get("signer_id") or r.get("signerId") or r.get("uploader") or ""

        # save NPZ in canonical SAKSHAM format
        out_path = accepted_dir / f"{gloss}_{uid}.npz"
        np.savez_compressed(
            out_path,
            left_hand=left_hand.astype(np.float32),
            right_hand=right_hand.astype(np.float32),
            left_present=left_present.astype(np.float32),
            right_present=right_present.astype(np.float32),
            left_confidence=left_conf.astype(np.float32),
            right_confidence=right_conf.astype(np.float32),
            timestamps=timestamps.astype(np.float32),
            gloss=gloss,
            signer_id=signer_id,
            uid=uid,
            video_path=video_path,
            frame_count=int(frame_count),
        )

        stats["success"] += 1
        per_gloss_counts[gloss] += 1

    # write simple stats
    report = out_dir / "extraction_stats.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"processed={stats['processed']}\n")
        f.write(f"success={stats['success']}\n")
        f.write(f"failed={stats['failed']}\n")
        f.write("per_gloss_counts:\n")
        for g, c in per_gloss_counts.most_common():
            f.write(f"{g},{c}\n")

    # write recommended vocab file
    vocab_path = out_dir / "recommended_v1.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        if hello_replacement:
            # replace HELLO in recommended list if present
            recommended = [hello_replacement if x == "HELLO" else x for x in recommended]
        for item in recommended:
            f.write(item + "\n")

    logging.info("Extraction complete. Stats: %s", stats)
    logging.info("Wrote recommended vocab to %s", vocab_path)
    logging.info("Wrote extraction stats to %s", report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
