"""
tools/generate_null_dataset.py
==============================
Generates synthetic and baseline MediaPipe Holistic (30, 225) sequences
for the 'null' (idle / non-signing) class.

Simulates:
1. Resting body pose with hands down / absent (0 hands, standard torso/head pose)
2. Resting hands on desk / lap with minor natural micro-movement
3. Random hand wandering / transitional trajectories across field of view
4. Neutral open palm / loose waving gestures that are not valid ASL signs

Output shape: (30, 225) float32 saved as .npz into dataset/processed/kaggle_holistic/null/
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("NullGenerator")

DEFAULT_OUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "kaggle_holistic" / "null"

SEQUENCE_LENGTH = 30
INPUT_FEATURES = 225


def generate_typical_pose() -> np.ndarray:
    """Generates realistic baseline 33-point body pose coordinates for a seated person."""
    pose = np.zeros((33, 3), dtype=np.float32)

    # Head / Face points (0..10) centered around (0.5, 0.25, 0.0)
    for i in range(11):
        pose[i] = [0.5 + np.random.uniform(-0.04, 0.04), 0.25 + np.random.uniform(-0.04, 0.04), np.random.uniform(-0.05, 0.05)]

    # Shoulders (11=left, 12=right)
    pose[11] = [0.62 + np.random.uniform(-0.02, 0.02), 0.45 + np.random.uniform(-0.02, 0.02), 0.0]
    pose[12] = [0.38 + np.random.uniform(-0.02, 0.02), 0.45 + np.random.uniform(-0.02, 0.02), 0.0]

    # Elbows (13=left, 14=right)
    pose[13] = [0.68 + np.random.uniform(-0.03, 0.03), 0.65 + np.random.uniform(-0.03, 0.03), 0.0]
    pose[14] = [0.32 + np.random.uniform(-0.03, 0.03), 0.65 + np.random.uniform(-0.03, 0.03), 0.0]

    # Wrists (15=left, 16=right)
    pose[15] = [0.65 + np.random.uniform(-0.04, 0.04), 0.85 + np.random.uniform(-0.04, 0.04), 0.0]
    pose[16] = [0.35 + np.random.uniform(-0.04, 0.04), 0.85 + np.random.uniform(-0.04, 0.04), 0.0]

    # Hips (23=left, 24=right)
    pose[23] = [0.58, 0.95, 0.0]
    pose[24] = [0.42, 0.95, 0.0]

    return pose


def generate_null_sample(sample_type: int) -> np.ndarray:
    """
    Generates a single (30, 225) null sequence based on type:
    0: Seated pose, no hands in frame
    1: Seated pose, hands resting near bottom / desk
    2: Smooth random hand drift across frame (transitional motion)
    3: Loose waving / fidgeting hand motion
    """
    seq = np.zeros((SEQUENCE_LENGTH, INPUT_FEATURES), dtype=np.float32)
    base_pose = generate_typical_pose()

    for t in range(SEQUENCE_LENGTH):
        # Add slight natural breathing / postural sway to pose
        current_pose = base_pose.copy()
        current_pose[:, 0] += np.sin(t * 0.2) * 0.005 + np.random.uniform(-0.002, 0.002, size=33)
        current_pose[:, 1] += np.cos(t * 0.2) * 0.003 + np.random.uniform(-0.002, 0.002, size=33)

        # Write pose to indices 126..224
        seq[t, 126:225] = current_pose.flatten()

        if sample_type == 0:
            # No hands in frame (left & right = 0)
            pass

        elif sample_type == 1:
            # Hands resting on table / bottom
            # Right hand (63..125)
            r_center = np.array([0.45 + np.random.uniform(-0.01, 0.01), 0.85 + np.random.uniform(-0.01, 0.01), 0.0])
            for i in range(21):
                seq[t, 63 + i * 3 : 63 + i * 3 + 3] = r_center + np.random.uniform(-0.03, 0.03, size=3)

        elif sample_type == 2:
            # Smooth random walk drift
            x_drift = 0.3 + 0.4 * (t / SEQUENCE_LENGTH) + np.sin(t * 0.3) * 0.05
            y_drift = 0.5 + np.cos(t * 0.4) * 0.1
            r_center = np.array([x_drift, y_drift, 0.0])
            for i in range(21):
                seq[t, 63 + i * 3 : 63 + i * 3 + 3] = r_center + np.random.uniform(-0.04, 0.04, size=3)

        elif sample_type == 3:
            # Left hand loose fidget
            x_drift = 0.6 + np.sin(t * 0.5) * 0.08
            y_drift = 0.6 + np.cos(t * 0.5) * 0.08
            l_center = np.array([x_drift, y_drift, 0.0])
            for i in range(21):
                seq[t, i * 3 : i * 3 + 3] = l_center + np.random.uniform(-0.04, 0.04, size=3)

    return seq


def generate_null_dataset(out_dir: Path, num_samples: int = 400):
    from src.sign_language.normalization import normalize_raw_225_sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Generating %d synthetic invariant-normalized null/idle samples in %s...", num_samples, out_dir)

    for i in range(num_samples):
        st = i % 4
        arr = generate_null_sample(st)
        norm_arr = normalize_raw_225_sequence(arr)
        out_path = out_dir / f"null_seq_{i:04d}.npz"
        np.savez_compressed(out_path, features=norm_arr, isometric_landmarks=norm_arr)

    log.info("Successfully generated %d normalized null sequences.", num_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic null/idle class samples.")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for null samples.")
    parser.add_argument("--count", type=int, default=400, help="Number of samples to generate.")
    args = parser.parse_args()

    generate_null_dataset(Path(args.dest), args.count)
