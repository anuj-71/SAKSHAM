"""
ONNX CNN-GRU sign language inference engine for SAKSHAM.
Runs real-time inference on sliding windows of (30, 225) holistic landmarks with:
1. Physics motion gating (delta >= 0.008 to run inference)
2. 0.4-second continuous hold-time confirmation (instant responsiveness)
3. 2.5-second hard timestamp cooldown between confirmations
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ONNXInferenceEngine")

SEQ_LEN = 30
FEATURE_DIM = 225            # 21 Left Hand (63) + 21 Right Hand (63) + 33 Pose (99)
MOTION_THRESHOLD = 0.008     # Mean absolute difference threshold to trigger inference (delta >= 0.008)
HOLD_TIME_SECONDS = 0.4      # Required continuous hold duration with confidence >= 0.75
MIN_CONFIDENCE = 0.75        # Minimum confidence threshold (0.75)
STABILITY_COOLDOWN = 2.5     # 2.5s hard timestamp cooldown between confirmed signs
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
        motion_threshold: float = MOTION_THRESHOLD,
        hold_time_seconds: float = HOLD_TIME_SECONDS,
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
        self.motion_threshold = motion_threshold
        self.hold_time_seconds = hold_time_seconds
        self.cooldown_seconds = cooldown_seconds

        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map: dict[str, int] = json.load(f)
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)

        # Sliding window buffer: stores (225,) numpy arrays
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=seq_len)

        # Physics Motion-based gating state
        self._prev_landmarks: Optional[np.ndarray] = None
        self.motion_score: float = 0.0

        # Continuous Hold Confirmation & Hard 2.5s Timestamp Cooldown
        self._hold_sign: Optional[str] = None
        self._hold_start_time: float = 0.0
        self._last_confirmed_time: float = 0.0

        # Raw / Live display state
        self._last_prediction: Optional[str] = None
        self._last_confidence: float = 0.0
        self._last_top3: list[tuple[str, float]] = []

        # Idle detection state
        self._last_hand_time: float = time.time()

        # Confirmed prediction state (public)
        self.current_prediction: Optional[str] = None
        self.current_confidence: float = 0.0
        self.current_top3: list[tuple[str, float]] = []

        logger.info(
            "ONNX CNN-GRU Engine loaded: %d classes, seq_len=%d, min_conf=%.2f, motion_gate=%.4f, hold=%.1fs, cooldown=%.1fs",
            self.num_classes, seq_len, min_confidence, motion_threshold, hold_time_seconds, cooldown_seconds,
        )

    def add_frame(self, hand_data: dict) -> dict:
        """
        Accepts one frame of MediaPipe holistic data, appends 225 coordinates to buffer,
        evaluates physics motion gate (delta < 0.008 skips inference), and runs CNN-GRU inference every frame.
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

        # Calculate landmark motion (mean absolute difference against previous frame)
        if self._prev_landmarks is not None:
            motion = float(np.mean(np.abs(frame_features - self._prev_landmarks)))
        else:
            motion = 0.0
        self._prev_landmarks = frame_features.copy()
        self.motion_score = motion

        # Only evaluate when buffer has accumulated enough temporal history
        if len(self.frame_buffer) < self.seq_len:
            return self._build_state()

        # Motion Gate: If hands/body are essentially still (< 0.008), skip inference entirely
        if motion < self.motion_threshold:
            self._hold_sign = None
            self._hold_start_time = 0.0
            self.current_prediction = None
            return self._build_state()

        # Run inference every frame
        self._run_inference()

        return self._build_state()

    def _reset_on_idle(self):
        """Clear frame buffer, motion state, and hold timer when hands/pose are absent."""
        self.frame_buffer.clear()
        self._prev_landmarks = None
        self.motion_score = 0.0
        self._hold_sign = None
        self._hold_start_time = 0.0
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_top3 = []
        self._last_prediction = None
        self._last_confidence = 0.0

    def get_state(self) -> dict:
        return self._build_state()

    def reset(self):
        self.frame_buffer.clear()
        self._prev_landmarks = None
        self.motion_score = 0.0
        self._hold_sign = None
        self._hold_start_time = 0.0
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_top3 = []
        self._last_prediction = None
        self._last_confidence = 0.0

    def get_raw_state(self) -> dict:
        now = time.time()
        hold_progress = 0.0
        if self._hold_sign and self._hold_start_time > 0.0:
            hold_progress = min(1.0, (now - self._hold_start_time) / self.hold_time_seconds)

        return {
            "raw_prediction": self._last_prediction,
            "raw_confidence": self._last_confidence,
            "raw_top3": list(self._last_top3),
            "hold_sign": self._hold_sign,
            "hold_progress": hold_progress,
            "motion_score": self.motion_score,
            "motion_threshold": self.motion_threshold,
            "is_moving": self.motion_score >= self.motion_threshold,
            "buffer_progress": len(self.frame_buffer),
            "buffer_capacity": self.seq_len,
        }

    def _extract_frame(self, hand_data: dict) -> np.ndarray:
        """Extracts 225 coordinates (21 left + 21 right + 33 pose landmarks x 3) from holistic data."""
        features = np.zeros(FEATURE_DIM, dtype=np.float32)

        left = hand_data.get("left_hand", {})
        right = hand_data.get("right_hand", {})
        pose = hand_data.get("pose", {})

        # Left Hand (0..62)
        if left.get("present") and left.get("landmarks"):
            lm_list = left["landmarks"]
            for idx, pt in enumerate(lm_list[:21]):
                if len(pt) >= 3:
                    features[idx * 3]     = pt[0]
                    features[idx * 3 + 1] = pt[1]
                    features[idx * 3 + 2] = pt[2]

        # Right Hand (63..125)
        if right.get("present") and right.get("landmarks"):
            lm_list = right["landmarks"]
            for idx, pt in enumerate(lm_list[:21]):
                if len(pt) >= 3:
                    offset = 63 + (idx * 3)
                    features[offset]     = pt[0]
                    features[offset + 1] = pt[1]
                    features[offset + 2] = pt[2]

        # Pose (126..224)
        if pose.get("present") and pose.get("landmarks"):
            lm_list = pose["landmarks"]
            for idx, pt in enumerate(lm_list[:33]):
                if len(pt) >= 3:
                    offset = 126 + (idx * 3)
                    features[offset]     = pt[0]
                    features[offset + 1] = pt[1]
                    features[offset + 2] = pt[2]

        return features

    def _run_inference(self):
        """Run ONNX CNN-GRU inference on the current (30, 225) buffer."""
        frames = np.stack(list(self.frame_buffer), axis=0).astype(np.float32)  # (30, 225)
        input_tensor = np.expand_dims(frames, axis=0)  # (1, 30, 225)

        try:
            logits = self.session.run(None, {self.input_name: input_tensor})[0]  # (1, num_classes)
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True))[0]
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

        now = time.time()

        # ── Check Hard 2.5-second Cooldown ──
        if (now - self._last_confirmed_time) < self.cooldown_seconds:
            self._hold_sign = None
            self._hold_start_time = 0.0
            self.current_prediction = None
            self._last_prediction = None
            self._last_confidence = 0.0
            return

        # ── Continuous 0.4s Hold Confirmation ──
        if best_confidence >= self.min_confidence:
            self._last_prediction = best_label
            self._last_confidence = best_confidence

            if self._hold_sign == best_label:
                # Continuing to hold the same sign
                elapsed_hold = now - self._hold_start_time
                if elapsed_hold >= self.hold_time_seconds:
                    # Confirmed after 0.4s continuous hold!
                    self.current_prediction = best_label
                    self.current_confidence = best_confidence
                    self.current_top3 = list(self._last_top3)
                    self._last_confirmed_time = now

                    # Reset hold state and clear buffer
                    self._hold_sign = None
                    self._hold_start_time = 0.0
                    self.frame_buffer.clear()
                    logger.info("Confirmed Sign: %s (held for %.2fs, conf=%.1f%%, cooldown 2.5s started)",
                                best_label, elapsed_hold, best_confidence * 100.0)
                else:
                    # Holding in progress (< 0.4s)
                    self.current_prediction = None
            else:
                # Started a new sign or sign changed -> start/reset 0.4s hold timer
                self._hold_sign = best_label
                self._hold_start_time = now
                self.current_prediction = None
        else:
            # Below 0.75 confidence -> reset hold timer
            self._hold_sign = None
            self._hold_start_time = 0.0
            self._last_prediction = None
            self._last_confidence = 0.0
            self.current_prediction = None

    def _build_state(self) -> dict:
        return {
            "prediction": self.current_prediction,
            "confidence": self.current_confidence,
            "top3": list(self.current_top3),
            "buffer_progress": len(self.frame_buffer),
            "buffer_capacity": self.seq_len,
        }
