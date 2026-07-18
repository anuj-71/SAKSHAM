"""Analyze V1 results and gather data for V2 plan."""
import numpy as np
from pathlib import Path
from collections import Counter
import json
import random

random.seed(42)
accepted = list(Path('cislr_data/landmarks/accepted').glob('*.npz'))

# Load V1 vocabulary
with open('cislr_data/processed/label_map.json') as f:
    v1 = json.load(f)

# ── HAND PRESENCE ──
print("=== HAND PRESENCE ANALYSIS ===")
sampled = random.sample(accepted, min(2000, len(accepted)))

lp_ratios = []
rp_ratios = []
both_ratios = []
no_hand_ratios = []

for p in sampled:
    with np.load(p) as d:
        lp = float(np.mean(d['left_present']))
        rp = float(np.mean(d['right_present']))
        lp_ratios.append(lp)
        rp_ratios.append(rp)
        both_ratios.append(min(lp, rp))
        no_hand_ratios.append(1.0 - max(lp, rp))

cats = {"both_high": 0, "left_only": 0, "right_only": 0, "sporadic": 0, "never": 0}
for lp, rp in zip(lp_ratios, rp_ratios):
    if lp > 0.5 and rp > 0.5:
        cats["both_high"] += 1
    elif lp > 0.5:
        cats["left_only"] += 1
    elif rp > 0.5:
        cats["right_only"] += 1
    elif lp > 0 or rp > 0:
        cats["sporadic"] += 1
    else:
        cats["never"] += 1

print(f"Total samples analyzed: {len(sampled)}")
print(f"Left  hand presence - mean: {np.mean(lp_ratios):.3f}")
print(f"Right hand presence - mean: {np.mean(rp_ratios):.3f}")
for k in ["both_high", "left_only", "right_only", "sporadic", "never"]:
    pct = cats[k] / len(sampled) * 100
    print(f"  {k}: {cats[k]} ({pct:.1f}%)")

# ── V1 VOCABULARY HAND PRESENCE ──
print(f"\n=== V1 VOCABULARY HAND PRESENCE ===")
v1_npz = [p for p in accepted if p.stem.split('_')[0].replace('#', 'N_A') in v1
          or any(p.stem.startswith(g.replace(' ', '_')) for g in v1)]
# simpler: read gloss from npz
v1_present = {"both": 0, "one": 0, "none": 0, "total": 0}
for p in accepted:
    with np.load(p) as d:
        gloss = str(d["gloss"])
        if gloss not in v1:
            continue
        v1_present["total"] += 1
        lp = float(np.mean(d['left_present']))
        rp = float(np.mean(d['right_present']))
        if lp > 0.5 and rp > 0.5:
            v1_present["both"] += 1
        elif lp > 0.5 or rp > 0.5:
            v1_present["one"] += 1
        else:
            v1_present["none"] += 1

print(f"V1 samples: {v1_present['total']}")
for k in ["both", "one", "none"]:
    pct = v1_present[k] / v1_present["total"] * 100 if v1_present["total"] else 0
    print(f"  {k}: {v1_present[k]} ({pct:.1f}%)")

# Check actual V1 test data hand presence
print(f"\n=== V1 TEST SET HAND PRESENCE ===")
import csv
with open('cislr_data/processed/test.csv') as f:
    reader = csv.DictReader(f)
    test_paths = [r['npz_path'] for r in reader]

test_correct = {"both": 0, "one": 0, "none": 0, "total": 0}
for path in test_paths:
    with np.load(path) as d:
        lp = float(np.mean(d['left_present']))
        rp = float(np.mean(d['right_present']))
        test_correct["total"] += 1
        if lp > 0.5 and rp > 0.5:
            test_correct["both"] += 1
        elif lp > 0.5 or rp > 0.5:
            test_correct["one"] += 1
        else:
            test_correct["none"] += 1
print(f"Test set samples: {test_correct['total']}")
for k in ["both", "one", "none"]:
    pct = test_correct[k] / test_correct["total"] * 100 if test_correct["total"] else 0
    print(f"  {k}: {test_correct[k]} ({pct:.1f}%)")


# ── V1 per-class split counts ──
print(f"\n=== V1 PER-CLASS SPLIT BREAKDOWN ===")
from collections import defaultdict
train_counts = Counter()
val_counts = Counter()
test_counts = Counter()
for split_name, counter in [('train.csv', train_counts), ('val.csv', val_counts), ('test.csv', test_counts)]:
    with open(f'cislr_data/processed/{split_name}') as f:
        for r in csv.DictReader(f):
            counter[r['label']] += 1

print(f"{'Label':25s} {'Train':>5s} {'Val':>5s} {'Test':>5s} {'Total':>5s}")
for g in sorted(v1.keys()):
    tr = train_counts.get(g, 0)
    va = val_counts.get(g, 0)
    te = test_counts.get(g, 0)
    print(f"{g:25s} {tr:5d} {va:5d} {te:5d} {tr+va+te:5d}")

# ── FRAME LENGTH DISTRIBUTION ──
print(f"\n=== FRAME LENGTH -> SEQ_LEN CHOICE ===")
frame_lens = []
for p in random.sample(accepted, min(500, len(accepted))):
    with np.load(p) as d:
        frame_lens.append(d['left_hand'].shape[0])
fls = sorted(frame_lens)
M = len(fls)
print(f"Frames: min={min(fls)}, median={fls[M//2]}, mean={sum(fls)/M:.1f}, max={max(fls)}")
for target in [24, 32, 48, 64, 96, 128]:
    truncated = sum(1 for f in fls if f > target)
    pct = truncated / M * 100
    print(f"  seq_len={target}: {truncated}/{M} videos truncated ({pct:.1f}%)")
# beyond median
padded_vs_truncated = sum(1 for f in fls if f <= 48)
print(f"  seq_len=48: {padded_vs_truncated}/{M} videos padded ({padded_vs_truncated/M*100:.1f}%) vs truncated")
padded_vs_truncated = sum(1 for f in fls if f <= 64)
print(f"  seq_len=64: {padded_vs_truncated}/{M} videos padded ({padded_vs_truncated/M*100:.1f}%) vs truncated")
