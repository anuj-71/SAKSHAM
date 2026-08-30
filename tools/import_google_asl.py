"""
tools/import_google_asl.py
==========================
Imports and filters Google ASL Signs (Kaggle Competition) dataset for SAKSHAM.
Extracts 15 target sign classes directly into SAKSHAM's processed dataset format:
  - 126 input features per frame: 21 left hand (x,y,z) + 21 right hand (x,y,z)
  - Fixed 30-frame temporal resampled sequences
  - Automatically generates dataset/metadata/manifest.csv and models/label_map.json

Usage:
  .venv\\Scripts\\python.exe tools/import_google_asl.py --source_dir path/to/asl-signs
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("GoogleASLImporter")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"
METADATA_DIR = DATASET_DIR / "metadata"
MODELS_DIR = PROJECT_ROOT / "models"

# 14 Core Conversational Classes present in Google ASL
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
    "water",
    "where",
    "yes",
]

# Alias map to handle dataset naming variations
CLASS_ALIASES: Dict[str, str] = {
    "hello": "hello",
    "bye": "bye",
    "goodbye": "bye",
    "yes": "yes",
    "no": "no",
    "please": "please",
    "thankyou": "thankyou",
    "thank_you": "thankyou",
    "thanks": "thankyou",
    "water": "water",
    "drink": "drink",
    "food": "food",
    "eat": "food",
    "hungry": "hungry",
    "home": "home",
    "sick": "sick",
    "where": "where",
    "sleep": "sleep",
}

SEQUENCE_LENGTH = 30
INPUT_FEATURES = 126  # 21 Left Hand (x,y,z) + 21 Right Hand (x,y,z)


def resample_sequence(frames: np.ndarray, target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Interpolates or resamples a temporal sequence of shape (T, 126) to exactly (target_length, 126).
    """
    t_in, d = frames.shape
    if t_in == 0:
        return np.zeros((target_length, d), dtype=np.float32)
    if t_in == target_length:
        return frames.astype(np.float32)

    # 1D linear interpolation along time axis
    orig_idx = np.linspace(0, 1, t_in)
    target_idx = np.linspace(0, 1, target_length)

    out = np.zeros((target_length, d), dtype=np.float32)
    for col in range(d):
        out[:, col] = np.interp(target_idx, orig_idx, frames[:, col])

    return out.astype(np.float32)


def process_parquet_file(parquet_path: Path) -> Optional[np.ndarray]:
    """
    Reads a Google ASL parquet file and extracts 126 hand coordinates (21 left + 21 right).
    Vectorized NumPy implementation for maximum speed.
    """
    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    except Exception as e:
        return None

    if "frame" not in df.columns or "type" not in df.columns:
        return None

    # Filter for hand landmarks only
    hands_df = df[df["type"].isin(["left_hand", "right_hand"])]
    if hands_df.empty:
        return None

    unique_frames = sorted(df["frame"].unique())
    num_frames = len(unique_frames)
    if num_frames == 0:
        return None

    frame_to_idx = {f: i for i, f in enumerate(unique_frames)}
    raw_features = np.zeros((num_frames, INPUT_FEATURES), dtype=np.float32)

    h_types = hands_df["type"].values
    h_lm = hands_df["landmark_index"].values.astype(int)
    h_frames = hands_df["frame"].map(frame_to_idx).values
    h_x = np.nan_to_num(hands_df["x"].values, nan=0.0).astype(np.float32)
    h_y = np.nan_to_num(hands_df["y"].values, nan=0.0).astype(np.float32)
    h_z = np.nan_to_num(hands_df["z"].values, nan=0.0).astype(np.float32)

    # Left hand landmarks (0..62: 21 * 3)
    l_mask = (h_types == "left_hand") & (h_lm >= 0) & (h_lm < 21)
    if np.any(l_mask):
        lf_idx = h_frames[l_mask]
        l_offset = h_lm[l_mask] * 3
        raw_features[lf_idx, l_offset]     = h_x[l_mask]
        raw_features[lf_idx, l_offset + 1] = h_y[l_mask]
        raw_features[lf_idx, l_offset + 2] = h_z[l_mask]

    # Right hand landmarks (63..125: 21 * 3)
    r_mask = (h_types == "right_hand") & (h_lm >= 0) & (h_lm < 21)
    if np.any(r_mask):
        rf_idx = h_frames[r_mask]
        r_offset = 63 + (h_lm[r_mask] * 3)
        raw_features[rf_idx, r_offset]     = h_x[r_mask]
        raw_features[rf_idx, r_offset + 1] = h_y[r_mask]
        raw_features[rf_idx, r_offset + 2] = h_z[r_mask]

    # Resample to fixed 30 frames
    return resample_sequence(raw_features, target_length=SEQUENCE_LENGTH)


def main():
    parser = argparse.ArgumentParser(description="Import 15 classes from Google ASL Signs dataset into SAKSHAM.")
    parser.add_argument("--source_dir", type=str, default="cislr_data",
                        help="Path to unzipped Google ASL dataset root (containing train.csv and train_landmark_files/)")
    parser.add_argument("--max_samples_per_class", type=int, default=400,
                        help="Maximum samples to import per class (default: 400 for balanced CPU training)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    csv_path = source_dir / "train.csv"

    if not csv_path.exists():
        # Try looking in subdirectories
        matches = list(source_dir.glob("**/train.csv"))
        if matches:
            csv_path = matches[0]
            source_dir = csv_path.parent
        else:
            log.error("train.csv not found in '%s'. Please specify --source_dir <path_to_dataset>", source_dir)
            sys.exit(1)

    log.info("Found train.csv at: %s", csv_path)

    # Prepare output directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Read manifest CSV
    df = pd.read_csv(csv_path)
    log.info("Total rows in source dataset: %d", len(df))

    # Match target classes
    df["canonical_sign"] = df["sign"].astype(str).str.lower().map(CLASS_ALIASES)
    filtered_df = df[df["canonical_sign"].isin(TARGET_CLASSES)].copy()

    log.info("Rows matching 15 target classes: %d", len(filtered_df))

    # Create label map
    label_map = {cls_name: idx for idx, cls_name in enumerate(sorted(TARGET_CLASSES))}
    label_map_path = MODELS_DIR / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    log.info("Saved label map to: %s (%d classes)", label_map_path, len(label_map))

    manifest_rows = []
    class_counts = {c: 0 for c in TARGET_CLASSES}
    signers_set = set()

    for _, row in filtered_df.iterrows():
        sign = row["canonical_sign"]
        if class_counts[sign] >= args.max_samples_per_class:
            continue

        rel_path = str(row["path"]).replace("\\", "/")
        parquet_file = source_dir / rel_path

        if not parquet_file.exists():
            continue

        features = process_parquet_file(parquet_file)
        if features is None or np.all(features == 0):
            continue

        sequence_id = str(row.get("sequence_id", f"{sign}_{class_counts[sign]}"))
        signer_id = str(row.get("participant_id", "SIGNER_01"))
        signers_set.add(signer_id)

        sample_filename = f"{sign}_{sequence_id}.npz"
        out_npz_path = PROCESSED_DIR / sample_filename

        # Save processed feature array
        np.savez_compressed(out_npz_path, isometric_landmarks=features, features=features)

        # 80/20 train/val split based on participant_id for true zero-leakage evaluation
        split = "val" if (hash(signer_id) % 5 == 0) else "train"

        manifest_rows.append({
            "file_path": sample_filename,
            "label": sign,
            "signer_id": signer_id,
            "source": "google_asl",
            "split": split,
        })

        class_counts[sign] += 1
        if len(manifest_rows) % 500 == 0 or len(manifest_rows) == len(filtered_df):
            log.info("Processed %4d samples...", len(manifest_rows))

    # Create label map from classes that actually have samples
    active_classes = sorted([cls_name for cls_name in TARGET_CLASSES if class_counts[cls_name] > 0])
    label_map = {cls_name: idx for idx, cls_name in enumerate(active_classes)}
    label_map_path = MODELS_DIR / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    # Write manifest.csv
    manifest_csv = METADATA_DIR / "manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "label", "signer_id", "source", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    log.info("=" * 60)
    log.info("IMPORT COMPLETED SUCCESSFULLY")
    log.info("Total samples processed: %d across %d signers", len(manifest_rows), len(signers_set))
    log.info("Class distribution (%d active classes):", len(active_classes))
    for cls_name in active_classes:
        log.info("  %-12s: %4d samples", cls_name, class_counts[cls_name])
    log.info("Manifest written to: %s", manifest_csv)
    log.info("Label map written to: %s (%d classes)", label_map_path, len(label_map))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
