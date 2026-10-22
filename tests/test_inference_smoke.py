"""Smoke test for inference pipeline."""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import pytest
from src.sign_language.engine import MODEL_PATH, LABEL_MAP_PATH
from src.sign_language.inference import ONNXInferenceEngine
from src.sign_language.engine import SignLanguageEngine
import numpy as np


def test_inference_smoke_pipeline():
    if not MODEL_PATH.exists():
        pytest.skip(
            f"model.onnx not found at {MODEL_PATH} - run tools/train.py first.",
        )

    # Check the ONNX engine loads
    eng = ONNXInferenceEngine(str(MODEL_PATH), str(LABEL_MAP_PATH))
    assert eng.num_classes > 0
    assert eng.label_map is not None

    # Test dummy frame
    state = eng.add_frame({
        'left_hand': {'present': True, 'landmarks': [(0.5, 0.5, 0.0)]*21},
        'right_hand': {'present': True, 'landmarks': [(0.5, 0.5, 0.0)]*21},
    })
    assert state["buffer_progress"] == 1

    # Fill buffer
    for _ in range(30):
        eng.add_frame({
            'left_hand': {'present': True, 'landmarks': [(0.5, 0.5, 0.0)]*21},
            'right_hand': {'present': True, 'landmarks': [(0.5, 0.5, 0.0)]*21},
        })
    state = eng.get_state()
    assert state["buffer_progress"] == 30

    # Check SignLanguageEngine initializes
    sl_engine = SignLanguageEngine()
    assert sl_engine.inference is not None


if __name__ == "__main__":
    test_inference_smoke_pipeline()
    print("All smoke tests passed!")
