"""
ONNX GRU sign language inference engine.

Matches training preprocessing from src.sign_language.training.preprocessing
and runs inference via ONNX Runtime on a sliding window of hand landmarks.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.sign_language.training.normalization import normalize_hand_frame
from src.sign_language.training.preprocessing import (
    preprocess_sample_for_gru,
)

logger = logging.getLogger(__name__)

SEQ_LEN = 64
FEATURE_DIM = 128
STABILITY_WINDOW = 12        # Number of recent predictions used for majority vote
STABILITY_THRESHOLD = 0.60   # Fraction of window that must agree (e.g. 60% = 8/12)
MIN_CONFIDENCE = 0.55         # Minimum model confidence to even consider a prediction
STABILITY_COOLDOWN = 2.5     # Seconds between confirmed signs
IDLE_TIMEOUT = 1.0            # Seconds of no-hand before buffer is cleared
PREDICTION_LOG_PATH = os.path.join("exports", "onnx_predictions.jsonl")


class ONNXInferenceEngine:
    """Accumulates a sliding window of landmarks and runs ONNX GRU inference."""

    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        seq_len: int = SEQ_LEN,
        stability_frames: int = STABILITY_WINDOW,
        cooldown_seconds: float = STABILITY_COOLDOWN,
    ):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.seq_len = seq_len

        with open(label_map_path) as f:
            self.label_map: dict[str, int] = json.load(f)
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)

        # Sliding window buffer: list of per-frame dicts
        self.frame_buffer: deque[dict] = deque(maxlen=seq_len)

        # Majority-vote stability filter state
        self.cooldown_seconds = cooldown_seconds
        self._vote_window: deque[str] = deque(maxlen=STABILITY_WINDOW)
        self._last_confirmed_time = 0.0
        self._last_prediction: Optional[str] = None
        self._last_confidence: float = 0.0
        self._last_top3: list[tuple[str, float]] = []

        # Idle detection state
        self._last_hand_time: float = time.time()

        # Latest full inference result (public)
        self.current_prediction: Optional[str] = None
        self.current_confidence: float = 0.0
        self.current_top3: list[tuple[str, float]] = []

        logger.info(
            "ONNXInferenceEngine loaded: %d classes, seq_len=%d, stability=%d frames",
            self.num_classes, seq_len, stability_frames,
        )

    # ── Public API ──────────────────────────────────────────────────────

    def add_frame(self, hand_data: dict) -> dict:
        """
        Accept one frame of MediaPipe hand data, append to buffer, and
        return the current prediction state dict.
        """
        left_present = bool(hand_data.get("left_hand", {}).get("present", False))
        right_present = bool(hand_data.get("right_hand", {}).get("present", False))
        hands_present = left_present or right_present

        # ── A1: Idle detection ────────────────────────────────────────────────
        # If hands disappeared for longer than IDLE_TIMEOUT, clear stale data
        if hands_present:
            self._last_hand_time = time.time()
        else:
            idle_duration = time.time() - self._last_hand_time
            if idle_duration >= IDLE_TIMEOUT:
                self._reset_on_idle()
                return self._build_state()

        frame = self._extract_frame(hand_data)
        self.frame_buffer.append(frame)

        # Only run inference when buffer is full
        if len(self.frame_buffer) < self.seq_len:
            return self._build_state()

        self._run_inference()
        return self._build_state()

    def _reset_on_idle(self):
        """Clear frame buffer and vote window when hands have been absent too long."""
        if self.frame_buffer or self._vote_window:
            logger.debug("Idle timeout — resetting GRU buffer and vote window.")
        self.frame_buffer.clear()
        self._vote_window.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_top3 = []
        self._last_prediction = None

    def get_state(self) -> dict:
        """Return current prediction state without adding a frame."""
        return self._build_state()

    def reset(self):
        """Clear the frame buffer and prediction state."""
        self.frame_buffer.clear()
        self._vote_window.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_top3 = []
        self._last_prediction = None

    # ── Internal ────────────────────────────────────────────────────────

    def _extract_frame(self, hand_data: dict) -> dict:
        """Convert a HandTracker output frame into training-format arrays."""
        left = hand_data.get("left_hand", {})
        right = hand_data.get("right_hand", {})

        left_lm = np.asarray(left.get("landmarks", []), dtype=np.float32)
        right_lm = np.asarray(right.get("landmarks", []), dtype=np.float32)

        left_present = bool(left.get("present", False))
        right_present = bool(right.get("present", False))

        # Pad to 21 landmarks if missing / incomplete
        if left_lm.shape[0] < 21:
            pad = np.zeros((21 - left_lm.shape[0], 3), dtype=np.float32)
            left_lm = np.concatenate([left_lm, pad], axis=0) if left_lm.size > 0 else pad
        if right_lm.shape[0] < 21:
            pad = np.zeros((21 - right_lm.shape[0], 3), dtype=np.float32)
            right_lm = np.concatenate([right_lm, pad], axis=0) if right_lm.size > 0 else pad

        # Apply same normalization as training
        left_norm = normalize_hand_frame(left_lm, left_present)
        right_norm = normalize_hand_frame(right_lm, right_present)

        return {
            "left_hand": left_norm,      # (21, 3)
            "right_hand": right_norm,     # (21, 3)
            "left_present": np.float32(left_present),
            "right_present": np.float32(right_present),
        }

    def _buffer_to_sample(self) -> dict:
        """Convert the accumulated deque of frames into a sample dict
        matching the NPZ schema that preprocess_sample_for_gru expects."""
        frames = list(self.frame_buffer)
        n = len(frames)

        left_hand = np.stack([f["left_hand"] for f in frames], axis=0)       # (T, 21, 3)
        right_hand = np.stack([f["right_hand"] for f in frames], axis=0)     # (T, 21, 3)
        left_present = np.array([f["left_present"] for f in frames], dtype=np.float32)
        right_present = np.array([f["right_present"] for f in frames], dtype=np.float32)

        return {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "left_present": left_present,
            "right_present": right_present,
        }

    def _run_inference(self):
        """Run ONNX inference on current buffer and update prediction state."""
        sample = self._buffer_to_sample()

        # Match training preprocessing exactly
        processed = preprocess_sample_for_gru(
            sample, target_length=self.seq_len, include_presence=True,
        )
        features = processed["features"]  # (SEQ_LEN, FEATURE_DIM)
        input_tensor = np.expand_dims(features, axis=0).astype(np.float32)  # (1, 64, 128)

        logits = self.session.run(None, {self.input_name: input_tensor})[0]  # (1, 10)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        probs = probs[0]  # (10,)

        # Top-3
        top3_idx = np.argsort(probs)[-3:][::-1]
        self._last_top3 = [
            (self.idx_to_label[int(i)], float(probs[i])) for i in top3_idx
        ]

        best_idx = int(top3_idx[0])
        best_label = self.idx_to_label[best_idx]
        best_confidence = float(probs[best_idx])

        self._last_prediction = best_label
        self._last_confidence = best_confidence

        # A2: Majority-vote stability filter
        self._apply_stability(best_label, best_confidence)

    def _apply_stability(self, label: str, confidence: float):
        """Majority-vote over recent predictions — much more robust than consecutive frames."""
        # Drop low-confidence frames from the vote
        if confidence < MIN_CONFIDENCE:
            return

        self._vote_window.append(label)

        if len(self._vote_window) < STABILITY_WINDOW:
            return  # Not enough data yet

        # Count votes
        from collections import Counter
        counts = Counter(self._vote_window)
        top_label, top_count = counts.most_common(1)[0]
        vote_fraction = top_count / len(self._vote_window)

        if vote_fraction >= STABILITY_THRESHOLD:
            now = time.time()
            if now - self._last_confirmed_time >= self.cooldown_seconds:
                self.current_prediction = top_label
                self.current_confidence = confidence
                self.current_top3 = self._last_top3
                self._last_confirmed_time = now
                logger.info(
                    "Sign confirmed: %s (conf=%.3f, vote=%.0f%%)",
                    top_label, confidence, vote_fraction * 100,
                )
                self._log_prediction()
                # Clear window after confirmation to prevent immediate re-fire
                self._vote_window.clear()

    def _log_prediction(self):
        """Append confirmed prediction to JSONL file for evaluation."""
        entry = {
            "timestamp": time.time(),
            "prediction": self.current_prediction,
            "confidence": round(self.current_confidence, 4),
            "top3": [(l, round(c, 4)) for l, c in self.current_top3],
            "buffer_size": len(self.frame_buffer),
        }
        os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
        try:
            with open(PREDICTION_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to log prediction: %s", e)

    def _build_state(self) -> dict:
        """Return the current observable prediction state."""
        return {
            "prediction": self.current_prediction,
            "confidence": self.current_confidence,
            "top3": self.current_top3,
            "buffer_filled": len(self.frame_buffer) >= self.seq_len,
            "buffer_progress": len(self.frame_buffer),
            "buffer_capacity": self.seq_len,
        }
