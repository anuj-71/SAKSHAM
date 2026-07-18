"""
SAKSHAM Dataset Configuration
Centralized configuration for dataset paths and validation settings.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(r"E:\gessture")
DATASET_ROOT = PROJECT_ROOT / "SAKSHAM_DATASET"

# Dataset files
TRAIN_CSV = DATASET_ROOT / "train_filtered.csv"
SIGN_MAP_JSON = DATASET_ROOT / "sign_map.json"
LANDMARK_FILES_DIR = DATASET_ROOT / "train_landmark_files"

# Output paths
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Validation settings
EXPECTED_SIGN_COUNT = 15
EXPECTED_SAMPLE_COUNT_APPROX = 5770
PARQUET_CHUNK_SIZE = 100  # For progress reporting

# Create output directories
REPORTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
