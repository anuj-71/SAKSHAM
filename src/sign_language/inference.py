"""
ONNX CNN-GRU sign language inference engine for SAKSHAM.
Runs real-time inference on sliding windows of (30, 126) hand landmarks.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ONNXInferenceEngine")

SEQ_LEN = 30
FEATURE_DIM = 225            # 21 Left Hand (63) + 21 Right Hand (63) + 33 Pose (99)
VOTING_BUFFER_SIZE = 10      # Collect last 10 frames of predictions
VOTING_CONFIRM_THRESHOLD = 7 # Confirm only if same class appears in >= 7 of 10 frames
MIN_CONFIDENCE = 0.75        # Minimum confidence threshold (idle/null filter)
STABILITY_COOLDOWN = 3.0     # 3.0s hard timestamp cooldown between confirmed signs
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
        voting_buffer_size: int = VOTING_BUFFER_SIZE,
        voting_confirm_threshold: int = VOTING_CONFIRM_THRESHOLD,
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

        # 10-Frame Voting Buffer and Hard 3.0s Timestamp Cooldown
        self.voting_buffer_size = voting_buffer_size
        self.voting_confirm_threshold = voting_confirm_threshold
        self.prediction_history: deque[str] = deque(maxlen=voting_buffer_size)
        self._vote_window: deque[np.ndarray] = deque(maxlen=voting_buffer_size)

        self.cooldown_seconds = cooldown_seconds
        self.inference_stride = 1  # Evaluate every frame for dense 10-frame voting
        self.frame_counter = 0
        self._last_confirmed_time: float = 0.0
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
            "ONNX CNN-GRU Engine loaded: %d classes, seq_len=%d, min_conf=%.2f, voting=%d/%d, cooldown=%.1fs",
            self.num_classes, seq_len, min_confidence, voting_confirm_threshold, voting_buffer_size, cooldown_seconds,
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
        """Clear frame buffer and voting history when hands/pose are absent."""
        self.frame_buffer.clear()
        self.prediction_history.clear()
        self._vote_window.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_top3 = []
        self._last_prediction = None

    def get_state(self) -> dict:
        return self._build_state()

    def reset(self):
        self.frame_buffer.clear()
        self.prediction_history.clear()
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
            "vote_window_size": len(self.prediction_history),
            "vote_window_capacity": self.voting_buffer_size,
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

        # Treat null or low-confidence as "null"
        if best_label == "null" or best_confidence < self.min_confidence:
            raw_prediction = "null"
            self._last_prediction = None
            self._last_confidence = 0.0
        else:
            raw_prediction = best_label
            self._last_prediction = best_label
            self._last_confidence = best_confidence

        # Record prediction into 10-frame voting history
        self.prediction_history.append(raw_prediction)
        self._vote_window.append(probs)

        # Apply 7/10 voting buffer & timestamp cooldown filter
        self._apply_voting_confirmation()

    def _apply_voting_confirmation(self):
        """
        Collects last 10 frames of predictions.
        Confirms a sign ONLY if the same class appears in >= 7 of 10 frames.
        Enforces a hard 3.0-second timestamp cooldown using time.time().
        """
        if len(self.prediction_history) < self.voting_buffer_size:
            self.current_prediction = None
            return

        # Count frequencies in the 10-frame window
        counts = Counter(self.prediction_history)
        most_common_label, count = counts.most_common(1)[0]

        # Must meet the 7/10 agreement threshold and cannot be 'null'
        if most_common_label == "null" or count < self.voting_confirm_threshold:
            self.current_prediction = None
            self.current_confidence = 0.0
            self.current_top3 = []
            return

        # ── EXACT COOLDOWN CODE BLOCK (Persistent time.time() check) ──
        now = time.time()
        elapsed = now - self._last_confirmed_time
        if elapsed < self.cooldown_seconds:
            # Under hard cooldown: block confirmation completely
            self.current_prediction = None
            self.current_confidence = 0.0
            self.current_top3 = []
            return

        # Passed 7/10 agreement AND 3.0s cooldown -> Confirm sign once!
        avg_probs = np.mean(list(self._vote_window), axis=0)
        c_idx = self.label_map.get(most_common_label, int(np.argmax(avg_probs)))
        confidence = float(avg_probs[c_idx])

        self.current_prediction = most_common_label
        self.current_confidence = confidence
        self.current_top3 = [
            (self.idx_to_label.get(int(i), f"class_{i}"), float(avg_probs[i]))
            for i in np.argsort(avg_probs)[-3:][::-1]
        ]
        self._last_confirmed_time = now

        # Flush buffers to avoid duplicate triggers from the same physical gesture
        self.prediction_history.clear()
        self._vote_window.clear()
        self.frame_buffer.clear()
        logger.info("Confirmed Sign: %s (%d/%d frames agreement, conf=%.1f%%, cooldown 3.0s started)",
                    most_common_label, count, self.voting_buffer_size, confidence * 100.0)

    def _build_state(self) -> dict:
        return {
            "prediction": self.current_prediction,
            "confidence": self.current_confidence,
            "top3": list(self.current_top3),
            "buffer_progress": len(self.frame_buffer),
            "buffer_capacity": self.seq_len,
        }
