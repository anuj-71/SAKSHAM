"""Full stability test for inference pipeline."""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from src.sign_language.inference import ONNXInferenceEngine
import numpy as np

import pytest
from pathlib import Path


def test_inference_stability_simulation():
    _MODEL = Path('models/model.onnx')
    _LABEL_MAP = Path('models/label_map.json')
    if not _MODEL.exists() or not _LABEL_MAP.exists():
        pytest.skip(f"model.onnx not found at {_MODEL} - run tools/train.py first.")

    eng = ONNXInferenceEngine(str(_MODEL), str(_LABEL_MAP), stability_frames=3)

    # Use different hand positions to get more varied predictions
    # Left hand open, right hand closed
    hand_data = {
        'left_hand': {'present': True, 'landmarks': [(0.3 + i*0.02, 0.4 + i*0.01, -0.01*i) for i in range(21)]},
        'right_hand': {'present': False, 'landmarks': [(0.0, 0.0, 0.0)]*21},
    }

    # Fill buffer to trigger stability
    state = None
    for i in range(68):
        state = eng.add_frame(hand_data)
        if state['prediction']:
            print(f'Frame {i+1}: CONFIRMED: {state["prediction"]} (conf={state["confidence"]:.3f})')
        elif i >= 63:
            print(f'Frame {i+1}: buffer full, running inference...')

    assert state is not None
    assert state['buffer_progress'] == 30


if __name__ == "__main__":
    test_inference_stability_simulation()
