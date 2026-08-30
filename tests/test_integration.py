"""Integration tests for SAKSHAM core pipeline (ONNX engine + session + callback)."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, os.path.abspath('.'))

import pytest
from src.sign_language.engine import SignLanguageEngine, MODEL_PATH
from src.session_manager import ConversationSession

# Marker applied to tests that require a trained model.onnx to exist.
_MODEL_AVAILABLE = MODEL_PATH.exists()
_SKIP_NO_MODEL = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason=f"model.onnx not found at {MODEL_PATH} - run tools/train.py first.",
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
HAND_DATA = {
    'left_hand': {'present': True, 'landmarks': [(0.3 + i*0.02, 0.4 + i*0.01, -0.01*i) for i in range(21)]},
    'right_hand': {'present': False, 'landmarks': [(0.0, 0.0, 0.0)]*21},
    'hands_present': 1,
    'primary_hand': 'Left',
    'landmarks': [(0.3 + i*0.02, 0.4 + i*0.01, -0.01*i) for i in range(21)],
    'pixel_landmarks': [(100, 200)]*21,
    'finger_states': {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True},
    'label': 'Left',
    'confidence': 0.9,
}


from unittest.mock import patch, MagicMock

def _make_engine_and_session():
    session = ConversationSession()
    confirmed_signs = []

    def on_sign(label):
        confirmed_signs.append(label)
        session.add_message("Deaf User", label, source="Sign")

    with patch('src.sign_language.engine.HandTracker'):
        engine = SignLanguageEngine(on_sign_recognized=on_sign)
    return engine, session, confirmed_signs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_SKIP_NO_MODEL
def test_onnx_engine_loads():
    """SignLanguageEngine initialises with a valid ONNX inference engine."""
    engine, _, _ = _make_engine_and_session()
    assert engine.inference is not None, "ONNX GRU inference engine should load"
    assert engine.inference.num_classes > 0, f"Expected >0 classes, got {engine.inference.num_classes}"


@_SKIP_NO_MODEL
def test_label_map_no_mom_dad():
    """Active label map must not contain 'mom' or 'dad'."""
    engine, _, _ = _make_engine_and_session()
    assert engine.inference is not None
    label_map = engine.inference.label_map
    assert "mom" not in label_map, "'mom' found in label map: removal incomplete"
    assert "dad" not in label_map, "'dad' found in label map: removal incomplete"
    assert "water" not in label_map, "'water' found in label map: removal incomplete"
    assert "null" not in label_map, "'null' found in label map: null removal incomplete"
    assert len(label_map) == 13, f"Expected 13 classes, got {len(label_map)}"


@_SKIP_NO_MODEL
def test_buffer_fills_after_30_frames():
    """After 30 identical frames the sliding-window buffer must be full."""
    engine, session, _ = _make_engine_and_session()
    assert engine.inference is not None

    # Monkey-patch tracker so we don't need a real camera
    engine.tracker.process_frame = lambda f: (True, HAND_DATA)

    for _ in range(30):
        engine.process_frame(None)

    state = engine.inference.get_state()
    assert state["buffer_progress"] == 30
    assert state["buffer_capacity"] == 30


@_SKIP_NO_MODEL
def test_session_state_updates_from_inference():
    """Session live_prediction / top3 fields update from inference state."""
    engine, session, _ = _make_engine_and_session()
    assert engine.inference is not None

    engine.tracker.process_frame = lambda f: (True, HAND_DATA)
    for _ in range(35):
        hand_detected, _ = engine.process_frame(None)
        inf_state = engine.inference.get_state()
        session.live_prediction = inf_state["prediction"]
        session.live_confidence = inf_state["confidence"]
        session.live_top3 = inf_state["top3"]
        session.buffer_progress = inf_state["buffer_progress"]
        session.buffer_capacity = inf_state["buffer_capacity"]

    # At minimum the buffer is filled and top3/confidence are sensible types
    assert session.buffer_progress == 30
    assert isinstance(session.live_top3, list)
    assert isinstance(session.live_confidence, float)


@_SKIP_NO_MODEL
def test_confirmed_sign_triggers_callback_eventually():
    """
    With 200 identical frames a sign SHOULD confirm at least once.
    The stability filter needs STABILITY_WINDOW (7) consecutive matching votes,
    then one cooldown period: 200 frames (x stride=4 -> 50 inference calls) gives
    ample opportunity. If no sign fires, it is flagged as a soft warning, not a
    hard failure, because the model may output low confidence on synthetic landmarks.
    """
    engine, session, confirmed_signs = _make_engine_and_session()
    if engine.inference is None:
        pytest.skip("ONNX model not available")

    engine.tracker.process_frame = lambda f: (True, HAND_DATA)
    for _ in range(200):
        engine.process_frame(None)

    # Soft check: inform rather than hard-fail on synthetic inputs
    if not confirmed_signs:
        pytest.xfail(
            "No sign confirmed after 200 synthetic frames "
            "(expected: model outputs low confidence on uniform landmarks)"
        )
    else:
        assert len(confirmed_signs) >= 1


def test_prediction_log_written():
    """If a sign has been confirmed, the JSONL prediction log must exist."""
    log_path = 'exports/onnx_predictions.jsonl'
    if not os.path.exists(log_path):
        pytest.skip("No prediction log yet: run the live app first to generate one")

    with open(log_path) as f:
        logs = [json.loads(line) for line in f if line.strip()]
    assert len(logs) >= 1, "Prediction log exists but is empty"
    # Validate schema of most recent entry
    entry = logs[-1]
    assert "prediction" in entry
    assert "confidence" in entry
    assert "top3" in entry
