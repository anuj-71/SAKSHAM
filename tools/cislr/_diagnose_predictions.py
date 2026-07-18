"""
Diagnose why STOP → GOOD/NAME/INDIA.

Simulate an open-palm STOP gesture and capture the full softmax distribution.
Compare with normalised landmarks from a real CISLR GOOD training sample.
"""
import json, numpy as np, csv, sys, os
sys.path.insert(0, os.path.abspath('.'))

from src.sign_language.inference import ONNXInferenceEngine
from src.sign_language.training.normalization import normalize_hand_frame
from src.sign_language.training.preprocessing import preprocess_sample_for_gru

eng = ONNXInferenceEngine(
    'cislr_data/model_v2/model.onnx',
    'cislr_data/model_v2/label_map.json',
    stability_frames=1,  # Immediate confirmation for diagnosis
    cooldown_seconds=0,
)

# ── Test 1: Simulate an OPEN PALM STOP gesture ──
# All fingers extended, palm facing camera, fingers pointing up
# MediaPipe returns 21 landmarks per hand
# Approximate coordinates for an open right hand, palm facing camera:
# x increases rightward, y increases downward, z increases away from camera
def open_palm_landmarks():
    """Generate synthetic landmarks for an open palm (STOP gesture)."""
    lm = []
    # Wrist (0)
    lm.append((0.5, 0.8, 0.0))
    # Thumb: CMC(1), MCP(2), IP(3), TIP(4) — extended to the side
    lm.append((0.45, 0.72, 0.01))
    lm.append((0.40, 0.65, 0.02))
    lm.append((0.36, 0.58, 0.03))
    lm.append((0.32, 0.52, 0.04))
    # Index: MCP(5), PIP(6), DIP(7), TIP(8) — pointing up
    lm.append((0.48, 0.65, 0.0))
    lm.append((0.48, 0.50, 0.0))
    lm.append((0.48, 0.42, 0.0))
    lm.append((0.48, 0.35, 0.0))
    # Middle: MCP(9), PIP(10), DIP(11), TIP(12) — pointing up
    lm.append((0.50, 0.64, 0.0))
    lm.append((0.50, 0.48, 0.0))
    lm.append((0.50, 0.39, 0.0))
    lm.append((0.50, 0.30, 0.0))
    # Ring: MCP(13), PIP(14), DIP(15), TIP(16) — pointing up
    lm.append((0.52, 0.65, 0.0))
    lm.append((0.52, 0.50, 0.0))
    lm.append((0.52, 0.42, 0.0))
    lm.append((0.52, 0.35, 0.0))
    # Pinky: MCP(17), PIP(18), DIP(19), TIP(20) — pointing up
    lm.append((0.54, 0.66, 0.0))
    lm.append((0.54, 0.54, 0.0))
    lm.append((0.54, 0.48, 0.0))
    lm.append((0.54, 0.42, 0.0))
    return lm

print("=" * 70)
print("DIAGNOSIS: STOP gesture -> model predictions")
print("=" * 70)

# Run 70 frames of STOP
hand_data = {
    'left_hand': {'present': False, 'landmarks': [(0.0, 0.0, 0.0)]*21},
    'right_hand': {'present': True, 'landmarks': open_palm_landmarks()},
    'hands_present': 1,
    'primary_hand': 'Right',
    'landmarks': open_palm_landmarks(),
    'pixel_landmarks': [(100, 200)]*21,
    'finger_states': {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True},
    'label': 'Right',
    'confidence': 0.9,
}

# First, manually probe the softmax for a single frame
print("\n--- Per-frame softmax distribution (frame 65 / full buffer) ---")
# Fill buffer
for i in range(64):
    eng.add_frame(hand_data)

# Now probe internal state by getting the feature vector
# and running it through ONNX manually
from src.sign_language.inference import preprocess_sample_for_gru
import onnxruntime as ort

# Build the exact same sample the engine uses
frame_buffer = list(eng.frame_buffer)
left_hand = np.stack([f["left_hand"] for f in frame_buffer], axis=0)
right_hand = np.stack([f["right_hand"] for f in frame_buffer], axis=0)
left_present = np.array([f["left_present"] for f in frame_buffer], dtype=np.float32)
right_present = np.array([f["right_present"] for f in frame_buffer], dtype=np.float32)

sample = {"left_hand": left_hand, "right_hand": right_hand,
          "left_present": left_present, "right_present": right_present}
processed = preprocess_sample_for_gru(sample, target_length=64, include_presence=True)
features = processed["features"]
input_tensor = np.expand_dims(features, axis=0).astype(np.float32)

logits = eng.session.run(None, {eng.input_name: input_tensor})[0]
probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
probs = probs[0]

# Full softmax distribution
print(f"\n  Full softmax distribution for STOP gesture:")
labels_sorted = sorted(eng.label_map.items(), key=lambda x: x[1])
for label, idx in labels_sorted:
    bar = "#" * int(probs[idx] * 100)
    print(f"  {label:15s} [{idx}] {probs[idx]:.4f} ({probs[idx]*100:.1f}%) {bar}")
print(f"\n  Top-1: {eng.idx_to_label[int(np.argmax(probs))]} ({np.max(probs):.2%})")
print(f"  Entropy: {-np.sum(probs * np.log(probs + 1e-10)):.3f} nats")
print(f"  Max prob: {np.max(probs):.3f}  (random baseline: {1/10:.3f})")

# ── Test 2: What does the model actually fire on? ──
print("\n" + "=" * 70)
print("PROBING: What does a REAL training sample of GOOD look like?")
print("=" * 70)

# Load a GOOD training sample
accepted_dir = 'cislr_data/landmarks/accepted'
good_files = [os.path.join(accepted_dir, f) for f in os.listdir(accepted_dir) 
              if f.startswith('GOOD_') and f.endswith('.npz')]
if good_files:
    with np.load(good_files[0]) as data:
        sample_real = {k: data[k] for k in data.files}
    processed_real = preprocess_sample_for_gru(sample_real, target_length=64, include_presence=True)
    features_real = processed_real["features"]
    input_tensor_real = np.expand_dims(features_real, axis=0).astype(np.float32)
    logits_real = eng.session.run(None, {eng.input_name: input_tensor_real})[0]
    probs_real = np.exp(logits_real) / np.sum(np.exp(logits_real), axis=1, keepdims=True)
    probs_real = probs_real[0]

    print(f"\n  Real GOOD sample ({os.path.basename(good_files[0])}):")
    for label, idx in labels_sorted:
        bar = "#" * int(probs_real[idx] * 100)
        print(f"  {label:15s} [{idx}] {probs_real[idx]:.4f} ({probs_real[idx]*100:.1f}%) {bar}")
    print(f"\n  Frame count: {sample_real['left_hand'].shape[0]}")
    print(f"  Left hand present (avg): {np.mean(sample_real['left_present']):.2f}")
    print(f"  Right hand present (avg): {np.mean(sample_real['right_present']):.2f}")

# ── Test 3: Compare feature vector structure ──
print("\n" + "=" * 70)
print("FEATURE ANALYSIS: What does the GRU actually see?")
print("=" * 70)

# The feature vector is 128-dim: 2 hands × 21 landmarks × 3 coords = 126, + 2 presence = 128
# Check which dimensions have signal
print(f"\n  STOP gesture feature vector stats:")
print(f"    Non-zero dims (any frame): {np.count_nonzero(np.abs(features).sum(axis=0) > 1e-6)} / {features.shape[1]}")

# Check presence signal
print(f"    Left presence frames: {np.sum(left_present > 0.5)} / {len(left_present)}")
print(f"    Right presence frames: {np.sum(right_present > 0.5)} / {len(right_present)}")

if good_files:
    print(f"\n  Real GOOD training sample feature vector stats:")
    print(f"    Non-zero dims (any frame): {np.count_nonzero(np.abs(features_real).sum(axis=0) > 1e-6)} / {features_real.shape[1]}")
    print(f"    Left presence frames: {np.sum(sample_real['left_present'] > 0.5)} / {sample_real['left_present'].shape[0]}")
    print(f"    Right presence frames: {np.sum(sample_real['right_present'] > 0.5)} / {sample_real['right_present'].shape[0]}")

# ── Test 4: Class prior check ──  
print("\n" + "=" * 70)
print("CLASS PRIOR: How many training samples per class?")
print("=" * 70)
train_counts = {}
with open('cislr_data/processed_v2/train.csv') as f:
    for r in csv.DictReader(f):
        train_counts[r['label']] = train_counts.get(r['label'], 0) + 1
total = sum(train_counts.values())
with open('cislr_data/reports/v2_training_report.json') as f:
    treport = json.load(f)
pca = treport.get('test_evaluation', {}).get('per_class_accuracy', {})
for label, idx in labels_sorted:
    ct = train_counts.get(label, 0)
    prior = ct / total
    learned = pca.get(label, {}).get('accuracy', 0) > 0
    learned_mark = " <-- LEARNED" if learned else ""
    print(f"  {label:15s}: {ct:2d} samples (prior={prior:.2f}){learned_mark}")
