"""
Path constants for the curated SAKSHAM dataset pipeline.

All paths point under dataset/ (not cislr_data/).
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT_DIR / "dataset"
RAW_VIDEOS_DIR = DATASET_DIR / "raw_videos"
LANDMARKS_DIR = DATASET_DIR / "landmarks"
PROCESSED_DIR = DATASET_DIR / "processed"
MODELS_DIR = DATASET_DIR / "models"

MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

MODEL_CHECKPOINT = MODELS_DIR / "gru_best.pt"
MODEL_ONNX = MODELS_DIR / "model.onnx"
MODEL_LABEL_MAP = MODELS_DIR / "label_map.json"
TRAINING_LOG = MODELS_DIR / "training_log.json"

def ensure_dirs():
    for d in [RAW_VIDEOS_DIR, LANDMARKS_DIR, PROCESSED_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
