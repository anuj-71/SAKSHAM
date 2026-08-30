"""
tools/extract_holistic_landmarks.py
===================================
Batch converts Google ASL Parquet files into SAKSHAM (30, 225) Holistic sequences:
  - 21 Left Hand (0..62)
  - 21 Right Hand (63..125)
  - 33 Pose Landmarks (126..224)
Total: 225 features per frame, 30 frames per sequence.
Excludes 'water' completely.

Saves directly into dataset/processed/kaggle_holistic/<sign_name>/
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HolisticExtractor")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_OUT_DIR = DATASET_DIR / "processed" / "kaggle_holistic"
MODELS_DIR = PROJECT_ROOT / "models"

# 13 Target Sign Classes (excluding water)
TARGET_CLASSES = [
    "bye",
    "drink",
    "food",
    "hello",
    "home",
    "hungry",
    "no",
    "please",
    "sick",
    "sleep",
    "thankyou",
    "where",
    "yes",
]

SEQUENCE_LENGTH = 30
INPUT_FEATURES = 225  # 21 Left (63) + 21 Right (63) + 33 Pose (99)


def resample_sequence(frames: np.ndarray, target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Interpolates sequence of shape (T, 225) to exactly (target_length, 225)."""
    t_in, d = frames.shape
    if t_in == 0:
        return np.zeros((target_length, d), dtype=np.float32)
    if t_in == target_length:
        return frames.astype(np.float32)

    orig_idx = np.linspace(0, 1, t_in)
    target_idx = np.linspace(0, 1, target_length)

    out = np.zeros((target_length, d), dtype=np.float32)
    for col in range(d):
        out[:, col] = np.interp(target_idx, orig_idx, frames[:, col])

    return out.astype(np.float32)


def process_parquet_file(parquet_path: Path) -> Optional[np.ndarray]:
    """
    Reads a Google ASL parquet file and extracts 225 coordinates:
    21 left hand (0..62) + 21 right hand (63..125) + 33 pose (126..224).
    Vectorized for maximum speed.
    """
    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    except Exception:
        return None

    if "frame" not in df.columns or "type" not in df.columns:
        return None

    # Filter for hands and pose landmarks
    relevant_df = df[df["type"].isin(["left_hand", "right_hand", "pose"])]
    if relevant_df.empty:
        return None

    unique_frames = sorted(df["frame"].unique())
    num_frames = len(unique_frames)
    if num_frames == 0:
        return None

    frame_to_idx = {f: i for i, f in enumerate(unique_frames)}
    raw_features = np.zeros((num_frames, INPUT_FEATURES), dtype=np.float32)

    types = relevant_df["type"].values
    lm_indices = relevant_df["landmark_index"].values.astype(int)
    f_indices = relevant_df["frame"].map(frame_to_idx).values
    x_vals = np.nan_to_num(relevant_df["x"].values, nan=0.0).astype(np.float32)
    y_vals = np.nan_to_num(relevant_df["y"].values, nan=0.0).astype(np.float32)
    z_vals = np.nan_to_num(relevant_df["z"].values, nan=0.0).astype(np.float32)

    # 1. Left Hand (0..62)
    l_mask = (types == "left_hand") & (lm_indices >= 0) & (lm_indices < 21)
    if np.any(l_mask):
        lf = f_indices[l_mask]
        l_offset = lm_indices[l_mask] * 3
        raw_features[lf, l_offset]     = x_vals[l_mask]
        raw_features[lf, l_offset + 1] = y_vals[l_mask]
        raw_features[lf, l_offset + 2] = z_vals[l_mask]

    # 2. Right Hand (63..125)
    r_mask = (types == "right_hand") & (lm_indices >= 0) & (lm_indices < 21)
    if np.any(r_mask):
        rf = f_indices[r_mask]
        r_offset = 63 + (lm_indices[r_mask] * 3)
        raw_features[rf, r_offset]     = x_vals[r_mask]
        raw_features[rf, r_offset + 1] = y_vals[r_mask]
        raw_features[rf, r_offset + 2] = z_vals[r_mask]

    # 3. Pose (126..224: 33 landmarks)
    p_mask = (types == "pose") & (lm_indices >= 0) & (lm_indices < 33)
    # Resample to 30 frames and apply 4-layer invariant normalization
    from src.sign_language.normalization import normalize_raw_225_sequence
    resampled = resample_sequence(raw_features, target_length=SEQUENCE_LENGTH)
    return normalize_raw_225_sequence(resampled)


def run_extraction(source_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = source_dir / "train.csv"
    if not csv_path.exists():
        log.error("train.csv not found in %s", source_dir)
        return

    df = pd.read_csv(csv_path)
    log.info("Loaded train.csv (%d total rows)", len(df))

    # Filter for 13 target classes (strictly no water)
    df_filtered = df[df["sign"].isin(TARGET_CLASSES)].copy()
    log.info("Filtered to %d samples across %d classes", len(df_filtered), len(TARGET_CLASSES))

    counts: Dict[str, int] = {c: 0 for c in TARGET_CLASSES}
    t0 = time.time()

    for idx, row in df_filtered.iterrows():
        sign = str(row["sign"])
        rel_path = str(row["path"])
        sequence_id = str(row["sequence_id"])

        parquet_path = source_dir / rel_path
        if not parquet_path.exists():
            continue

        arr = process_parquet_file(parquet_path)
        if arr is None:
            continue

        sign_dir = output_dir / sign
        sign_dir.mkdir(parents=True, exist_ok=True)

        out_path = sign_dir / f"seq_{sequence_id}.npz"
        np.savez_compressed(out_path, features=arr, isometric_landmarks=arr)
        counts[sign] += 1

        if sum(counts.values()) % 500 == 0:
            elapsed = time.time() - t0
            rate = sum(counts.values()) / max(elapsed, 0.001)
            log.info("Processed %d / %d samples (%.1f samples/sec)...",
                     sum(counts.values()), len(df_filtered), rate)

    total_saved = sum(counts.values())
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("EXTRACTION COMPLETE in %.1fs: %d samples saved", elapsed, total_saved)
    for c in TARGET_CLASSES:
        log.info("  - %-12s: %d samples", c, counts[c])
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 225-dim Holistic landmarks from Google ASL parquet files.")
    parser.add_argument("--source_dir", type=str, default="asl_data", help="Source folder containing train.csv and parquet files.")
    parser.add_argument("--dest_dir", type=str, default="dataset/processed/kaggle_holistic", help="Output directory.")
    args = parser.parse_args()

    run_extraction(Path(args.source_dir), Path(args.dest_dir))
