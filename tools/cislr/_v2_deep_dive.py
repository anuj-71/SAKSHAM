"""
Deep-dive into hand detection quality for MONDAY (most frequent, 0% usable).
Check if ANY frames have hand detections.
"""
import numpy as np
from pathlib import Path

accepted = Path('cislr_data/landmarks/accepted')
mondays = sorted(accepted.glob('MONDAY_*.npz'))

print(f"Found {len(mondays)} MONDAY NPZ files")
print()

for mp in mondays:
    with np.load(mp) as d:
        lp = d['left_present']
        rp = d['right_present']
        lc = d['left_confidence']
        rc = d['right_confidence']
        lh = d['left_hand']
        rh = d['right_hand']
        total_frames = d['frame_count']
        actual_frames = lh.shape[0]

        # Count frames with any hand detection
        any_hand = (lp > 0) | (rp > 0)
        both_hand = (lp > 0) & (rp > 0)
        no_hand = (lp == 0) & (rp == 0)

        # Check if hand landmarks have non-zero values
        lh_nonzero = np.any(np.abs(lh).sum(axis=(1,2)) > 1e-6)
        rh_nonzero = np.any(np.abs(rh).sum(axis=(1,2)) > 1e-6)

        print(f"{mp.name:50s} frames={actual_frames:3d} (reported={total_frames:3d}) "
              f"any_hand={any_hand.sum():3d}/{actual_frames} "
              f"both={both_hand.sum():3d} "
              f"no_hand={no_hand.sum():3d} "
              f"mean_lp={np.mean(lp):.3f} "
              f"lh_nonzero={lh_nonzero} "
              f"rh_nonzero={rh_nonzero}")

# Also check the 'best' sample in terms of detection
print()
print("=== Checking HAND DATA content (landmark coordinates) ===")
# Pick a MONDAY sample with most detections
best_monday = mondays[0]
with np.load(best_monday) as d:
    print(f"Sample: {best_monday.name}")
    print(f"left_present unique values: {np.unique(np.round(d['left_present'], 2))}")
    print(f"right_present unique values: {np.unique(np.round(d['right_present'], 2))}")
    # Get a frame where left is present
    lp_idx = np.where(d['left_present'] > 0)[0]
    rp_idx = np.where(d['right_present'] > 0)[0]
    print(f"Left present frames: {len(lp_idx)}")
    print(f"Right present frames: {len(rp_idx)}")
    if len(lp_idx) > 0:
        f = lp_idx[0]
        print(f"Left hand frame {f}: min={d['left_hand'][f].min():.4f}, max={d['left_hand'][f].max():.4f}, mean={d['left_hand'][f].mean():.4f}")
        print(f"  Coord range: x[{d['left_hand'][f,:,0].min():.3f}-{d['left_hand'][f,:,0].max():.3f}] "
              f"y[{d['left_hand'][f,:,1].min():.3f}-{d['left_hand'][f,:,1].max():.3f}] "
              f"z[{d['left_hand'][f,:,2].min():.3f}-{d['left_hand'][f,:,2].max():.3f}]")
    if len(rp_idx) > 0:
        f = rp_idx[0]
        print(f"Right hand frame {f}: min={d['right_hand'][f].min():.4f}, max={d['right_hand'][f].max():.4f}")
    # Check if absent frames have all-zero landmarks
    absent_frames = np.where(d['left_present'] == 0)[0]
    if len(absent_frames) > 0:
        f = absent_frames[0]
        all_zero = np.allclose(d['left_hand'][f], 0, atol=1e-6)
        print(f"Absent frame {f}: left_hand all_zero={all_zero}")

# Now check CAT (best usable class)
print()
print("=== CAT (best usable class) ===")
cats = sorted(accepted.glob('CAT_*.npz'))
for cp in cats:
    with np.load(cp) as d:
        lp = np.mean(d['left_present'])
        rp = np.mean(d['right_present'])
        any_hand = ((d['left_present'] > 0) | (d['right_present'] > 0)).sum()
        print(f"{cp.name:50s} frames={d['left_hand'].shape[0]:3d} "
              f"lp={lp:.2f} rp={rp:.2f} any_hand_frames={any_hand}")
