"""Smoke test for inference pipeline."""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import pytest
from src.sign_language.engine import MODEL_PATH, LABEL_MAP_PATH
from src.sign_language.inference import ONNXInferenceEngine
import numpy as np

print(f'Model exists: {MODEL_PATH.exists()}')
print(f'Label map exists: {LABEL_MAP_PATH.exists()}')
print('All imports OK')

# Skip entire module gracefully when no model has been trained yet.
# Once tools/train.py produces models/model.onnx, this guard becomes inactive.
if not MODEL_PATH.exists():
    pytest.skip(
        f"model.onnx not found at {MODEL_PATH} — run tools/train.py first.",
        allow_module_level=True,
    )

# Check the ONNX engine loads
eng = ONNXInferenceEngine(str(MODEL_PATH), str(LABEL_MAP_PATH))
print(f'Engine loaded: {eng.num_classes} classes')
print(f'Label map: {eng.label_map}')

# Test dummy frame
state = eng.add_frame({
    'left_hand': {'present': True, 'landmarks': [(0.5, 0.5, 0.0)]*21},
    'right_hand': {'present': True, 'landmarks': [(0.5, 0.5, 0.0)]*21},
})
print(f'State after 1 frame: buffer={state["buffer_progress"]}/64')

# Fill buffer
for _ in range(63):
    eng.add_frame({
        'left_hand': {'present': True, 'landmarks': [(0.5, 0.5, 0.0)]*21},
        'right_hand': {'present': True, 'landmarks': [(0.5, 0.5, 0.0)]*21},
    })
state = eng.get_state()
print(f'State after 64 frames: pred={state["prediction"]}, '
      f'top3={[(l, f"{c:.3f}") for l,c in state["top3"]]}')

# Check SignLanguageEngine initializes
from src.sign_language.engine import SignLanguageEngine
sl_engine = SignLanguageEngine()
print(f'SL Engine inference loaded: {sl_engine.inference is not None}')
print('All smoke tests passed!')
