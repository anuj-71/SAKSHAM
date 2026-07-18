import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable
from collections import deque

from src.sign_language.tracker import HandTracker
from src.sign_language.dataset_manager import DatasetManager
from src.sign_language.inference import ONNXInferenceEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = Path(__file__).resolve().parents[2] / "dataset" / "models" / "model.onnx"
LABEL_MAP_PATH = Path(__file__).resolve().parents[2] / "dataset" / "models" / "label_map.json"

class SignLanguageEngine:
    """
    Core engine for processing hand tracking data and recognizing sign language.
    Uses ONNX GRU model for inference with stability filtering.
    """
    def __init__(self, on_sign_recognized: Optional[Callable[[str], None]] = None):
        self.tracker = HandTracker()
        self.dataset_manager = DatasetManager()
        self.on_sign_recognized = on_sign_recognized
        
        self.inference = self._init_or_none()
        
        # Legacy fields for dataset recording compatibility
        self.sequence_buffer = deque(maxlen=30)
        self.sign_cooldowns: Dict[str, float] = {}
        self.cooldown_seconds = 2.5
        self.confidence_threshold = 0.75
        self.current_candidate = None
        self.candidate_frames = 0
        self.frames_to_confirm = 5
        self._last_notified_sign: Optional[str] = None

    def _init_or_none(self) -> Optional[ONNXInferenceEngine]:
        try:
            if MODEL_PATH.exists() and LABEL_MAP_PATH.exists():
                engine = ONNXInferenceEngine(
                    model_path=str(MODEL_PATH),
                    label_map_path=str(LABEL_MAP_PATH),
                )
                logging.info("ONNX GRU inference engine loaded (%s, %d classes)",
                             MODEL_PATH.name, engine.num_classes)
                return engine
            else:
                logging.warning("ONNX model not found at %s — inference disabled", MODEL_PATH)
                return None
        except Exception as e:
            logging.warning("Failed to load ONNX inference engine: %s", e)
            return None

    @staticmethod
    def _buffer_hand(hand: Dict) -> Dict:
        return {
            "present": bool(hand.get("present")),
            "landmarks": hand.get("landmarks", []),
            "finger_states": hand.get("finger_states", {}),
        }

    def process_frame(self, frame) -> tuple[bool, Dict]:
        """
        Process a single video frame. Extracts landmarks, updates sequence buffer,
        and runs ONNX inference when enough frames accumulated.
        """
        success, hand_data = self.tracker.process_frame(frame)
        
        if success and hand_data:
            # 1. Update legacy temporal buffer (for dataset recording)
            self.sequence_buffer.append({
                "timestamp": time.time(),
                "primary_hand": hand_data.get("primary_hand"),
                "primary_landmarks": hand_data.get("landmarks", []),
                "primary_finger_states": hand_data.get("finger_states", {}),
                "left_hand": self._buffer_hand(hand_data.get("left_hand", {})),
                "right_hand": self._buffer_hand(hand_data.get("right_hand", {})),
            })
            
            # 2. Add to dataset if recording
            if self.dataset_manager.is_recording:
                self.dataset_manager.add_frame_data(hand_data)
                
            # 3. ONNX GRU inference (if model loaded)
            if self.inference is not None:
                state = self.inference.add_frame(hand_data)
                # Fire callback only once per new confirmed prediction
                new_pred = state["prediction"]
                # Filter out unknown class (non-sign gestures)
                if new_pred == "unknown":
                    new_pred = None
                if new_pred and new_pred != self._last_notified_sign and self.on_sign_recognized:
                    self._last_notified_sign = new_pred
                    self.on_sign_recognized(new_pred)
                elif new_pred is None:
                    self._last_notified_sign = None
            else:
                # Fallback: legacy heuristic
                self._attempt_recognition(hand_data)
            
        return success, hand_data

    def _attempt_recognition(self, hand_data: Dict):
        """
        Attempts to recognize a sign from the current frame or sequence buffer.
        """
        current_time = time.time()
        
        # Simple Heuristic Recognition for Demo Vocabulary:
        # hello, yes, no, thank_you, help, stop, water

        finger_states = hand_data.get("finger_states", {})
        landmarks = hand_data.get("landmarks", [])

        if not finger_states or not landmarks:
            return

        thumb = finger_states.get("thumb", False)
        index = finger_states.get("index", False)
        middle = finger_states.get("middle", False)
        ring = finger_states.get("ring", False)
        pinky = finger_states.get("pinky", False)

        detected_sign = None
        confidence = 0.0

        if thumb and index and middle and ring and pinky:
            y_base = landmarks[0][1]
            y_middle_tip = landmarks[12][1]
            if (y_base - y_middle_tip) > 0.4:
                detected_sign = "stop"
                confidence = 0.85
            else:
                detected_sign = "hello"
                confidence = 0.80

        elif thumb and not index and not middle and not ring and not pinky:
            detected_sign = "yes"
            confidence = 0.90

        elif not thumb and index and not middle and not ring and not pinky:
            detected_sign = "no"
            confidence = 0.75

        elif thumb and index and middle and not ring and not pinky:
            detected_sign = "water"
            confidence = 0.85

        elif not thumb and not index and not middle and not ring and not pinky:
            detected_sign = "help"
            confidence = 0.80

        if detected_sign in ("hello", "stop") and len(self.sequence_buffer) > 10:
            past_landmarks = self.sequence_buffer[-10].get("primary_landmarks", [])
            if past_landmarks:
                past_wrist_y = past_landmarks[0][1]
                current_wrist_y = landmarks[0][1]
                if (current_wrist_y - past_wrist_y) > 0.1:
                    detected_sign = "thank_you"
                    confidence = 0.88

        # --- State Machine: Detected -> Candidate -> Confirmed ---
        if detected_sign and confidence >= self.confidence_threshold:
            if detected_sign == self.current_candidate:
                self.candidate_frames += 1
            else:
                self.current_candidate = detected_sign
                self.candidate_frames = 1
                
            if self.candidate_frames >= self.frames_to_confirm:
                # --- Confirmed State ---
                confirmed_sign = self.current_candidate
                
                # Cooldown check (per sign)
                last_time = self.sign_cooldowns.get(confirmed_sign, 0.0)
                if current_time - last_time >= self.cooldown_seconds:
                    # Update cooldown
                    self.sign_cooldowns[confirmed_sign] = current_time
                    
                    logging.info(f"Sign Confirmed: {confirmed_sign} (Conf: {confidence:.2f})")
                    
                    # Dataset Logging for confirmed sign
                    self._log_confirmed_sign(confirmed_sign, confidence, hand_data)
                    
                    if self.on_sign_recognized:
                        self.on_sign_recognized(confirmed_sign)
                        
                # Reset candidate after confirmation or cooldown skip
                self.current_candidate = None
                self.candidate_frames = 0
        else:
            # If sign is lost or changed, reset candidate state
            self.current_candidate = None
            self.candidate_frames = 0

    def _log_confirmed_sign(self, label: str, confidence: float, hand_data: Dict):
        """Logs a confirmed sign to a standalone file for dataset training."""
        import json, os
        log_file = os.path.join(self.dataset_manager.data_dir, "confirmed_signs_log.jsonl")
        
        entry = {
            "timestamp": time.time(),
            "label": label,
            "confidence": confidence,
            "hand_label": hand_data.get("label", "Unknown"),
            "primary_landmarks": [coord for pt in hand_data.get("landmarks", []) for coord in pt],
            "left_hand_present": hand_data.get("left_hand", {}).get("present", False),
            "right_hand_present": hand_data.get("right_hand", {}).get("present", False),
            "left_landmarks": [coord for pt in hand_data.get("left_hand", {}).get("landmarks", []) for coord in pt],
            "right_landmarks": [coord for pt in hand_data.get("right_hand", {}).get("landmarks", []) for coord in pt],
        }
        
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logging.error(f"Failed to log confirmed sign: {e}")

    # Dataset pass-through methods
    def start_recording(self, label: str, signer_id: str = "SIGNER_01"):
        self.dataset_manager.start_recording(label, signer_id)
        
    def stop_recording(self):
        return self.dataset_manager.stop_recording()

    def accept_recording(self):
        return self.dataset_manager.accept_current_clip()

    def reject_recording(self, reason: str = "Rejected"):
        return self.dataset_manager.reject_current_clip(reason)

    def has_review_clip(self) -> bool:
        return self.dataset_manager.has_review_clip()

    def get_review_summary(self):
        return self.dataset_manager.get_review_summary()
        
    def export_dataset(self, format: str = "json"):
        return self.dataset_manager.export_dataset(format)
