"""Test full SAKSHAM integration (engine + session + rendering)."""
import sys, os, json
sys.path.insert(0, os.path.abspath('.'))

from src.sign_language.engine import SignLanguageEngine
from src.session_manager import ConversationSession

# Test 1: SignLanguageEngine loads with ONNX
session = ConversationSession()
confirmed_signs = []

def on_sign(label):
    confirmed_signs.append(label)
    session.add_message("Deaf User", label, source="Sign")

sl_engine = SignLanguageEngine(on_sign_recognized=on_sign)
assert sl_engine.inference is not None, "ONNX engine should be loaded"
print("Test 1 PASS: Engine initializes with ONNX")

# Test 2: Simulate frames  
hand_data = {
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
sl_engine.tracker.process_frame = lambda f: (True, hand_data)

# Process frames (simulate main loop)
for i in range(68):
    hand_detected, hd = sl_engine.process_frame(None)
    inf_state = sl_engine.inference.get_state()
    session.live_prediction = inf_state["prediction"]
    session.live_confidence = inf_state["confidence"]
    session.live_top3 = inf_state["top3"]
    session.buffer_progress = inf_state["buffer_progress"]
    session.buffer_capacity = inf_state["buffer_capacity"]

print(f"Test 2 PASS: Processed 68 frames")
print(f"  live_prediction: {session.live_prediction}")
print(f"  live_confidence: {session.live_confidence:.3f}")
print(f"  live_top3: {[(l, f'{c:.3f}') for l, c in session.live_top3]}")
print(f"  buffer: {session.buffer_progress}/{session.buffer_capacity}")

# Test 3: Confirmed sign appears in conversation
print(f"  confirmed_signs: {confirmed_signs}")
assert len(confirmed_signs) >= 1, "Should have at least 1 confirmed sign"
print(f"  conversation messages: {len(session.messages)}")
assert len(session.messages) >= 1, "Messages should exist"

# Test 4: Prediction log exists
log_path = 'exports/onnx_predictions.jsonl'
assert os.path.exists(log_path), "Prediction log should exist"
with open(log_path) as f:
    logs = [json.loads(l) for l in f if l.strip()]
assert len(logs) >= 1
print(f"  prediction log entries: {len(logs)}")
for log in logs:
    print(f"    {log['prediction']} (conf={log['confidence']:.3f})")

print()
print("ALL TESTS PASSED - SAKSHAM ONNX integration complete!")
print(f"  Engine: ONNX GRU ({sl_engine.inference.num_classes} classes)")
print(f"  Buffer: {sl_engine.inference.seq_len} frames sliding window")
print(f"  Stability: {sl_engine.inference.stability_frames} frames")
print(f"  Cooldown: {sl_engine.inference.cooldown_seconds}s")
print(f"  Logging: predictions written to {log_path}")
