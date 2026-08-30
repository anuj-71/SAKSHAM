"""
ONNX CNN-GRU sign language inference engine for SAKSHAM.
Runs real-time inference on sliding windows of (30, 126) hand landmarks.
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

logger = logging.getLogger("ONNXInferenceEngine")

SEQ_LEN = 30
FEATURE_DIM = 225            # 21 Left Hand (63) + 21 Right Hand (63) + 33 Pose (99)
STABILITY_WINDOW = 2         # Instant 2-frame trigger (fast responsiveness)
STABILITY_THRESHOLD = 0.65   # Probability threshold to confirm a sign
MIN_CONFIDENCE = 0.65        # Minimum confidence threshold (idle/null filter)
STABILITY_COOLDOWN = 2.0     # 2.0s hard cooldown between confirmed signs
IDLE_TIMEOUT = 0.8           # Seconds of no-hands before clearing buffer
PREDICTION_LOG_PATH = os.path.join("exports", "onnx_predictions.jsonl")


class ONNXInferenceEngine:
    """Accumulates a sliding window of (30, 225) landmarks and runs ONNX CNN-GRU inference."""

    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        seq_len: int = SEQ_LEN,
        min_confidence: float = MIN_CONFIDENCE,
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
        self.min_confidence = min_confidence

        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map: dict[str, int] = json.load(f)
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)

        # Sliding window buffer: stores (225,) numpy arrays
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=seq_len)

        # Stability filter state
        self.cooldown_seconds = cooldown_seconds
        self.inference_stride = 2  # Run inference every 2 frames for ultra-smooth responsiveness
        self.frame_counter = 0
        self._vote_window: deque[np.ndarray] = deque(maxlen=stability_frames)
        self._last_confirmed_time = 0.0
        self._last_prediction: Optional[str] = None
        self._last_confidence: float = 0.0
        self._last_top3: list[tuple[str, float]] = []

        # Idle detection state
        self._last_hand_time: float = time.time()

        # Confirmed prediction state (public)
        self.current_prediction: Optional[str] = None
        self.current_confidence: float = 0.0
        self.current_top3: list[tuple[str, float]] = []

        self.stability_frames = stability_frames

        logger.info(
            "ONNX CNN-GRU Engine loaded: %d classes, seq_len=%d, min_conf=%.2f, cooldown=%.1fs",
            self.num_classes, seq_len, min_confidence, cooldown_seconds,
        )

    def add_frame(self, hand_data: dict) -> dict:
        """
        Accepts one frame of MediaPipe holistic data, appends 225 coordinates to buffer,
        and runs CNN-GRU inference when buffer is full.
        """
        left_present = bool(hand_data.get("left_hand", {}).get("present", False))
        right_present = bool(hand_data.get("right_hand", {}).get("present", False))
        pose_present = bool(hand_data.get("pose", {}).get("present", False))
        detected = left_present or right_present or pose_present

        # Idle detection
        if detected:
            self._last_hand_time = time.time()
        else:
            idle_duration = time.time() - self._last_hand_time
            if idle_duration >= IDLE_TIMEOUT:
                self._reset_on_idle()
                return self._build_state()

        frame_features = self._extract_frame(hand_data)
        self.frame_buffer.append(frame_features)
        self.frame_counter += 1

        # Only run inference when buffer is full
        if len(self.frame_buffer) < self.seq_len:
            return self._build_state()

        if self.frame_counter % self.inference_stride == 0:
            self._run_inference()

        return self._build_state()

    def _reset_on_idle(self):
        """Clear frame buffer and vote window when hands/pose are absent."""
        self.frame_buffer.clear()
        self._vote_window.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_top3 = []
        self._last_prediction = None

    def get_state(self) -> dict:
        return self._build_state()

    def reset(self):
        self.frame_buffer.clear()
        self._vote_window.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_top3 = []
        self._last_prediction = None

    def get_raw_state(self) -> dict:
        return {
            "raw_prediction": self._last_prediction,
            "raw_confidence": self._last_confidence,
            "raw_top3": list(self._last_top3),
            "vote_window_size": len(self._vote_window),
            "vote_window_capacity": self.stability_frames,
            "buffer_progress": len(self.frame_buffer),
            "buffer_capacity": self.seq_len,
        }

    def _extract_frame(self, hand_data: dict) -> np.ndarray:
        """Extracts and normalizes 225 coordinates using body-centric invariant normalizer."""
        from src.sign_language.normalization import normalize_holistic_frame

        left = hand_data.get("left_hand", {})
        right = hand_data.get("right_hand", {})
        pose = hand_data.get("pose", {})

        left_lms = np.array(left["landmarks"], dtype=np.float32) if left.get("present") and left.get("landmarks") else None
        right_lms = np.array(right["landmarks"], dtype=np.float32) if right.get("present") and right.get("landmarks") else None
        pose_lms = np.array(pose["landmarks"], dtype=np.float32) if pose.get("present") and pose.get("landmarks") else None

        return normalize_holistic_frame(left_lms, right_lms, pose_lms)

    def _run_inference(self):
        """Run clean ambidextrous ONNX CNN-GRU inference on the current (30, 225) buffer."""
        frames = np.stack(list(self.frame_buffer), axis=0).astype(np.float32)  # (30, 225)
        input_tensor = np.expand_dims(frames, axis=0)  # (1, 30, 225)

        # Mirrored frames for left/right hand ambidextrous symmetry
        mirr_frames = frames.copy()
        mirr_frames[:, 0:63] = frames[:, 63:126]
        mirr_frames[:, 63:126] = frames[:, 0:63]
        for col_idx in range(0, FEATURE_DIM, 3):
            mirr_frames[:, col_idx] = -mirr_frames[:, col_idx]
        mirr_tensor = np.expand_dims(mirr_frames, axis=0)

        try:
            logits_orig = self.session.run(None, {self.input_name: input_tensor})[0]
            logits_mirr = self.session.run(None, {self.input_name: mirr_tensor})[0]

            exp_orig = np.exp(logits_orig - np.max(logits_orig, axis=1, keepdims=True))
            probs_orig = (exp_orig / np.sum(exp_orig, axis=1, keepdims=True))[0]

            exp_mirr = np.exp(logits_mirr - np.max(logits_mirr, axis=1, keepdims=True))
            probs_mirr = (exp_mirr / np.sum(exp_mirr, axis=1, keepdims=True))[0]

            # Ensembled neural probabilities across left & right handedness
            probs = np.maximum(probs_orig, probs_mirr)
            probs = probs / np.sum(probs)
        except Exception as e:
            logger.error("ONNX inference failed: %s", e)
            return

        # Top-3 ranking
        top3_idx = np.argsort(probs)[-3:][::-1]
        self._last_top3 = [
            (self.idx_to_label.get(int(i), f"class_{i}"), float(probs[i])) for i in top3_idx
        ]

        best_idx = int(top3_idx[0])
        best_label = self.idx_to_label.get(best_idx, "unknown")
        best_confidence = float(probs[best_idx])

        if best_label == "null" or best_confidence < self.min_confidence:
            self._last_prediction = None
            self._last_confidence = 0.0
        else:
            self._last_prediction = best_label
            self._last_confidence = best_confidence

        # Soft voting stability filter
        self._apply_stability(probs)

    def _apply_stability(self, probs: np.ndarray):
        self._vote_window.append(probs)

        if len(self._vote_window) < self.stability_frames:
            return

        avg_probs = np.mean(list(self._vote_window), axis=0)
        best_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[best_idx])
        label = self.idx_to_label.get(best_idx, "unknown")

        now = time.time()

        # Fix 2: Idle / Null detection — if model predicts 'null' OR top confidence < 0.75, treat as no sign
        if label == "null" or confidence < self.min_confidence:
            self.current_prediction = None
            self.current_confidence = 0.0
            self.current_top3 = []
            return

        # Fix 1: Hard 3.0-second cooldown — block all confirmations during cooldown period
        if (now - self._last_confirmed_time) < self.cooldown_seconds:
            self.current_prediction = None
            self.current_confidence = 0.0
            self.current_top3 = []
            return

        # Confirm sign
        self.current_prediction = label
        self.current_confidence = confidence
        self.current_top3 = [
            (self.idx_to_label.get(int(i), f"class_{i}"), float(avg_probs[i]))
            for i in np.argsort(avg_probs)[-3:][::-1]
        ]
        self._last_confirmed_time = now
        self._vote_window.clear()
        self.frame_buffer.clear()
        logger.info("Confirmed Sign: %s (confidence=%.1f%%)", label, confidence * 100.0)

    def _build_state(self) -> dict:
        return {
            "prediction": self.current_prediction,
            "confidence": self.current_confidence,
            "top3": list(self.current_top3),
            "buffer_progress": len(self.frame_buffer),
            "buffer_capacity": self.seq_len,
        }
