import unittest
import sys
import os
import pytest

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config

# src.gesture_engine was removed in Phase 1 — replaced by ONNXInferenceEngine.
# These heuristic-gesture tests are obsolete; skip rather than error.
pytest.skip(
    "src.gesture_engine was removed in Phase 1 ONNX migration — tests obsolete",
    allow_module_level=True,
)

class TestGestureEngine(unittest.TestCase):
    def setUp(self):
        self.engine = GestureEngine()

    def test_resolve_static_gesture_closed_fist(self):
        # All fingers folded
        finger_states = {
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False
        }
        # Mock landmarks (normalized). We don't care about coordinates except for PINCH scale.
        # Wrist at (0,0,0), Middle MCP at (0, 0.2, 0). Hand scale = 0.2.
        landmarks = [(0.0, 0.0, 0.0)] * 21
        landmarks[0] = (0.0, 0.0, 0.0)
        landmarks[9] = (0.0, 0.2, 0.0)  # Hand scale is 0.2
        
        # Set thumb tip (4) and index tip (8) far apart to prevent PINCH
        landmarks[4] = (0.5, 0.0, 0.0)
        landmarks[8] = (0.0, 0.5, 0.0)
        
        gesture = self.engine._resolve_static_gesture(landmarks, finger_states, 1.0)
        self.assertEqual(gesture, "CLOSED_FIST")

    def test_resolve_static_gesture_open_palm(self):
        # All fingers extended
        finger_states = {
            "thumb": True,
            "index": True,
            "middle": True,
            "ring": True,
            "pinky": True
        }
        landmarks = [(0.0, 0.0, 0.0)] * 21
        landmarks[0] = (0.0, 0.0, 0.0)
        landmarks[9] = (0.0, 0.2, 0.0)  # Hand scale is 0.2
        landmarks[4] = (0.5, 0.0, 0.0)
        landmarks[8] = (0.0, 0.5, 0.0)
        
        gesture = self.engine._resolve_static_gesture(landmarks, finger_states, 1.0)
        self.assertEqual(gesture, "OPEN_PALM")

    def test_resolve_static_gesture_point(self):
        # Index extended, others folded
        finger_states = {
            "thumb": False,
            "index": True,
            "middle": False,
            "ring": False,
            "pinky": False
        }
        landmarks = [(0.0, 0.0, 0.0)] * 21
        landmarks[0] = (0.0, 0.0, 0.0)
        landmarks[9] = (0.0, 0.2, 0.0)  # Hand scale is 0.2
        landmarks[4] = (0.5, 0.0, 0.0)
        landmarks[8] = (0.0, 0.5, 0.0)
        
        gesture = self.engine._resolve_static_gesture(landmarks, finger_states, 1.0)
        self.assertEqual(gesture, "POINT")

    def test_resolve_static_gesture_peace(self):
        # Index and Middle extended, others folded
        finger_states = {
            "thumb": False,
            "index": True,
            "middle": True,
            "ring": False,
            "pinky": False
        }
        landmarks = [(0.0, 0.0, 0.0)] * 21
        landmarks[0] = (0.0, 0.0, 0.0)
        landmarks[9] = (0.0, 0.2, 0.0)  # Hand scale is 0.2
        landmarks[4] = (0.5, 0.0, 0.0)
        landmarks[8] = (0.0, 0.5, 0.0)
        
        gesture = self.engine._resolve_static_gesture(landmarks, finger_states, 1.0)
        self.assertEqual(gesture, "PEACE_SIGN")

    def test_resolve_static_gesture_thumbs_up(self):
        # Thumb extended, others folded
        finger_states = {
            "thumb": True,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False
        }
        landmarks = [(0.0, 0.0, 0.0)] * 21
        landmarks[0] = (0.0, 0.0, 0.0)
        landmarks[9] = (0.0, 0.2, 0.0)  # Hand scale = 0.2
        
        # Verify thumb tip is higher than thumb knuckle on y axis (y is smaller)
        # Thumb tip (4), Thumb base (2)
        landmarks[4] = (0.0, 0.1, 0.0)
        landmarks[2] = (0.0, 0.3, 0.0)
        
        # Distance between 4 and 8 must be large to avoid PINCH
        landmarks[8] = (0.5, 0.5, 0.0)
        
        gesture = self.engine._resolve_static_gesture(landmarks, finger_states, 1.0)
        self.assertEqual(gesture, "THUMBS_UP")

    def test_resolve_static_gesture_pinch(self):
        # Thumb and Index tip are close, other finger states don't override it
        finger_states = {
            "thumb": True,
            "index": True,
            "middle": True,
            "ring": True,
            "pinky": True
        }
        landmarks = [(0.0, 0.0, 0.0)] * 21
        landmarks[0] = (0.0, 0.0, 0.0)
        landmarks[9] = (0.0, 0.2, 0.0)  # Hand scale is 0.2
        
        # Pinch distance between 4 and 8 is small (0.005, which is < 0.05 * 0.2 = 0.01)
        landmarks[4] = (0.1, 0.1, 0.0)
        landmarks[8] = (0.105, 0.1, 0.0)
        
        gesture = self.engine._resolve_static_gesture(landmarks, finger_states, 0.05)
        self.assertEqual(gesture, "PINCH")

if __name__ == '__main__':
    unittest.main()
