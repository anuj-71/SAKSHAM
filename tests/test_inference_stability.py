"""Full stability test for inference pipeline."""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from src.sign_language.inference import ONNXInferenceEngine
import numpy as np

import pytest
from pathlib import Path

_MODEL = Path('models/model.onnx')
if not _MODEL.exists():
    pytest.skip(f"model.onnx not found at {_MODEL} — run tools/train.py first.",
                allow_module_level=True)

eng = ONNXInferenceEngine(str('models/model.onnx'),
                           str('models/label_map.json'),
                           stability_frames=3)

# Use different hand positions to get more varied predictions
# Left hand open, right hand closed
hand_data = {
    'left_hand': {'present': True, 'landmarks': [(0.3 + i*0.02, 0.4 + i*0.01, -0.01*i) for i in range(21)]},
    'right_hand': {'present': False, 'landmarks': [(0.0, 0.0, 0.0)]*21},
}

# Fill buffer to trigger stability
for i in range(68):
    state = eng.add_frame(hand_data)
    if state['prediction']:
        print(f'Frame {i+1}: CONFIRMED: {state["prediction"]} (conf={state["confidence"]:.3f})')
        print(f'  Top3: {[(l, f"{c:.3f}") for l,c in state["top3"]]}')
    elif i >= 63:
        print(f'Frame {i+1}: buffer full, running inference... top3: {[(l, f"{c:.3f}") for l,c in state["top3"]]}')

print()
print('Final state:', state)

# Read prediction log
import json
log_path = 'exports/onnx_predictions.jsonl'
if os.path.exists(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    print(f'\nPrediction log entries: {len(lines)}')
    for line in lines:
        print(json.loads(line))
