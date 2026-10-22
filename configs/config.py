"""
SAKSHAM Project Configuration
Dataset-agnostic paths for the clean, fresh-start layout.

Layout:
  dataset/raw/         - raw source data (videos, images, etc.)
  dataset/processed/   - extracted isometric .npz landmark files
  dataset/metadata/    - manifest.csv + label_map.json (source of truth for classes)
  models/              - model.onnx + label_map.json (live inference artefacts)
  exports/             - ONNX prediction logs, debug exports
  reports/             - evaluation reports, confusion matrices
  outputs/             - misc script outputs
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset layout
DATASET_ROOT       = PROJECT_ROOT / "dataset"
RAW_DATA_DIR       = DATASET_ROOT / "raw"
PROCESSED_DATA_DIR = DATASET_ROOT / "processed"
METADATA_DIR       = DATASET_ROOT / "metadata"

# Metadata files (filled once classes are finalised)
MANIFEST_CSV       = METADATA_DIR / "manifest.csv"
LABEL_MAP_JSON     = METADATA_DIR / "label_map.json"   # vocabulary definition

# Model artefacts (written by train.py, read by engine.py)
MODELS_DIR         = PROJECT_ROOT / "models"
MODEL_ONNX         = MODELS_DIR / "model.onnx"
INFERENCE_LABEL_MAP = MODELS_DIR / "label_map.json"

# Misc outputs
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Ensure output dirs exist at import time
for _d in (REPORTS_DIR, OUTPUTS_DIR, EXPORTS_DIR, MODELS_DIR,
           RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)
