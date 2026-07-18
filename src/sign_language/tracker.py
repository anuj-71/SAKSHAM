import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import src.config.settings as config


# Compatibility: some mediapipe installs expose the solutions module directly
# as `mp.solutions`, others place it under `mediapipe.python.solutions`.
def _locate_mediapipe_solutions():
    """Robustly locate the mediapipe 'solutions' module across packaging variants.

    Strategy:
    - If `mp` has attribute `solutions`, return it.
    - Otherwise, walk `mp.__path__` using `pkgutil.walk_packages` and try to import
      any submodule whose final component is 'solutions' (e.g. mediapipe.python.solutions).
    - As a fallback, try a few known alternative import paths.
    Returns the imported solutions module or None.
    """
    # Direct attribute is best
    if hasattr(mp, "solutions"):
        return mp.solutions

    # Walk package submodules to find any 'solutions' package
    try:
        import pkgutil, importlib
        if hasattr(mp, "__path__"):
            # iterate over all submodules under mediapipe
            for finder, name, ispkg in pkgutil.walk_packages(path=mp.__path__, prefix="mediapipe."):
                # look for modules that end with '.solutions' or are exactly 'mediapipe.solutions'
                if name.endswith(".solutions") or name == "mediapipe.solutions":
                    try:
                        mod = importlib.import_module(name)
                        return mod
                    except Exception:
                        continue
    except Exception:
        pass

    # Try a few conventional fallbacks
    try:
        import importlib
        mod = importlib.import_module("mediapipe.python.solutions")
        return mod
    except Exception:
        pass

    try:
        mod = importlib.import_module("mediapipe.solutions")
        return mod
    except Exception:
        pass

    return None


# Acquire solutions namespace once
_MP_SOLUTIONS = _locate_mediapipe_solutions()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HandTracker:
    """
    Wraps Google MediaPipe Hands tracking. Detects hand landmarks,
    extracts coordinates, and determines finger extension states.
    """
    def __init__(self):
        if _MP_SOLUTIONS is None:
            logging.error("MediaPipe 'solutions' namespace could not be located. Check your mediapipe installation.")
            raise RuntimeError("mediapipe.solutions not found")

        self.mp_hands = _MP_SOLUTIONS.hands
        self.mp_draw = _MP_SOLUTIONS.drawing_utils
        
        # Initialize MediaPipe Hands model with specified confidence thresholds
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # MediaPipe Landmark connection mapping
        self.connections = self.mp_hands.HAND_CONNECTIONS
        logging.info("MediaPipe Hands initialized with confidence thresholds >= 0.8.")

    @staticmethod
    def _empty_hand(label: str) -> Dict:
        return {
            "label": label,
            "present": False,
            "confidence": 0.0,
            "landmarks": [],
            "pixel_landmarks": [],
            "finger_states": {},
            "raw_mp_landmarks": None,
        }

    def process_frame(self, frame: cv2.Mat) -> Tuple[bool, Dict]:
        """
        Processes a single BGR image frame.
        Returns a tuple of (hand_detected, hand_data_dictionary).
        """
        # Downsample the frame for MediaPipe tracking if specified
        h, w, _ = frame.shape
        
        if config.TRACKING_FRAME_WIDTH > 0 and config.TRACKING_FRAME_HEIGHT > 0:
            small_bgr = cv2.resize(frame, (config.TRACKING_FRAME_WIDTH, config.TRACKING_FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
            rgb_frame = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        results = self.hands.process(rgb_frame)
        
        left_hand = self._empty_hand("Left")
        right_hand = self._empty_hand("Right")

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                confidence = handedness.classification[0].score

                raw_landmarks = []
                pixel_landmarks = []
                for lm in landmarks.landmark:
                    raw_landmarks.append((lm.x, lm.y, lm.z))
                    pixel_landmarks.append((int(lm.x * w), int(lm.y * h)))

                finger_states = self._get_finger_states(raw_landmarks, label)
                hand_entry = {
                    "label": label,
                    "present": True,
                    "confidence": confidence,
                    "landmarks": raw_landmarks,
                    "pixel_landmarks": pixel_landmarks,
                    "finger_states": finger_states,
                    "raw_mp_landmarks": landmarks,
                }

                if label == "Left":
                    if (not left_hand["present"]) or confidence >= left_hand["confidence"]:
                        left_hand = hand_entry
                elif label == "Right":
                    if (not right_hand["present"]) or confidence >= right_hand["confidence"]:
                        right_hand = hand_entry

        hands_present = int(left_hand["present"]) + int(right_hand["present"])
        if hands_present == 0:
            return False, {}

        primary_hand = right_hand if right_hand["present"] else left_hand
        hand_data = {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "hands_present": hands_present,
            "primary_hand": primary_hand["label"],
            # Keep primary-hand aliases so the heuristic pipeline still works.
            "label": primary_hand["label"],
            "confidence": primary_hand["confidence"],
            "landmarks": primary_hand["landmarks"],
            "pixel_landmarks": primary_hand["pixel_landmarks"],
            "finger_states": primary_hand["finger_states"],
            "raw_mp_landmarks": primary_hand["raw_mp_landmarks"],
        }
        return True, hand_data

    def _get_finger_states(self, landmarks: List[Tuple[float, float, float]], handedness: str) -> Dict[str, bool]:
        """
        Analyzes landmarks to determine if individual fingers are extended (True) or folded (False).
        Returns a dictionary mapping: 'thumb', 'index', 'middle', 'ring', 'pinky' -> bool
        """
        states = {}
        
        # Helper function to calculate Euclidean distance in 2D space
        def dist_2d(p1_idx: int, p2_idx: int) -> float:
            x1, y1 = landmarks[p1_idx][0], landmarks[p1_idx][1]
            x2, y2 = landmarks[p2_idx][0], landmarks[p2_idx][1]
            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
        # 1. Thumb State
        # Robust rotation-independent heuristic: compare distance between thumb tip (4) and pinky base (17)
        # to distance between thumb base MCP (2) and pinky base (17).
        # Also compare thumb tip to index knuckle (5).
        # If thumb tip is extended, it should be further from the pinky MCP than the thumb knuckle.
        d_tip_pinky = dist_2d(4, 17)
        d_mcp_pinky = dist_2d(2, 17)
        
        # In addition, check if the thumb is extended horizontally away from the hand
        # If palm is facing camera: Right hand thumb is on the left (x4 < x2), Left hand thumb is on the right (x4 > x2)
        # We can also check if knuckle 5 is to the right of knuckle 17 to determine palm orientation.
        knuckles_right = landmarks[5][0] > landmarks[17][0]
        
        if knuckles_right: # Palm facing camera for Right hand, or back of hand for Left hand
            is_extended_horiz = landmarks[4][0] < landmarks[3][0] if handedness == "Right" else landmarks[4][0] > landmarks[3][0]
        else: # Back of hand for Right hand, or palm facing camera for Left hand
            is_extended_horiz = landmarks[4][0] > landmarks[3][0] if handedness == "Right" else landmarks[4][0] < landmarks[3][0]
            
        # Thumb is extended if the distance check passes OR the horizontal extension check passes
        states["thumb"] = (d_tip_pinky > d_mcp_pinky * 1.1) or is_extended_horiz
        
        # 2. Four Fingers (Index, Middle, Ring, Pinky)
        # Extended if the tip is above the PIP joint (y-coordinate is smaller in image space)
        # We also check the DIP joint to ensure the finger is straight.
        # Index: Tip (8), PIP (6), MCP (5)
        states["index"] = landmarks[8][1] < landmarks[6][1]
        
        # Middle: Tip (12), PIP (10), MCP (9)
        states["middle"] = landmarks[12][1] < landmarks[10][1]
        
        # Ring: Tip (16), PIP (14), MCP (13)
        states["ring"] = landmarks[16][1] < landmarks[14][1]
        
        # Pinky: Tip (20), PIP (18), MCP (17)
        states["pinky"] = landmarks[20][1] < landmarks[18][1]
        
        return states

    def get_hand_bbox(self, pixel_landmarks: List[Tuple[int, int]], padding: int = 15) -> Tuple[int, int, int, int]:
        """Calculates the bounding box (x_min, y_min, x_max, y_max) around the hand landmarks."""
        xs = [pt[0] for pt in pixel_landmarks]
        ys = [pt[1] for pt in pixel_landmarks]
        
        x_min = max(0, min(xs) - padding)
        y_min = max(0, min(ys) - padding)
        x_max = max(x_min, max(xs) + padding)
        y_max = max(y_min, max(ys) + padding)
        
        return x_min, y_min, x_max, y_max
