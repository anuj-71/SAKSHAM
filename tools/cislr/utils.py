"""
Shared utilities for the CISLR data preparation pipeline.
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# ── Directory layout ────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parents[2]   # e:/gessture
CISLR_DATA_DIR  = ROOT_DIR / "cislr_data"
RAW_VIDEOS_DIR  = CISLR_DATA_DIR / "raw_videos"
RAW_FRAMES_DIR  = CISLR_DATA_DIR / "raw_frames"
LANDMARKS_DIR   = CISLR_DATA_DIR / "landmarks"
REPORTS_DIR     = CISLR_DATA_DIR / "reports"
MODEL_DIR       = CISLR_DATA_DIR / "model"

DATASET_CSV     = CISLR_DATA_DIR / "dataset.csv"
DATASET_JSON    = CISLR_DATA_DIR / "dataset.json"
LABEL_FREQ_CSV  = CISLR_DATA_DIR / "label_frequency.csv"

# ── Target vocabulary ────────────────────────────────────────────────────────
TARGET_LABELS = ["SEE YOU TOMORROW", "THANK YOU", "YES", "NO", "HELP", "WATER", "STOP"]

# ── Temporal window ──────────────────────────────────────────────────────────
WINDOW_SIZE = 5          # consecutive frames per sample
LANDMARKS_PER_FRAME = 21 # MediaPipe hand landmark count
COORDS_PER_LANDMARK = 3  # x, y, z
FEATURES_PER_FRAME  = LANDMARKS_PER_FRAME * COORDS_PER_LANDMARK  # 63
FEATURE_DIM         = WINDOW_SIZE * FEATURES_PER_FRAME            # 315


def ensure_dirs():
    """Create all required output directories."""
    for d in [CISLR_DATA_DIR, RAW_VIDEOS_DIR, RAW_FRAMES_DIR,
              LANDMARKS_DIR, REPORTS_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_video_resolver(dataset_root=None):
    """
    Return (video_root, VideoPathResolver) for a CISLR dataset root.

    Future extraction scripts should use this helper rather than hard-coding
    a single directory level.
    """
    from tools.cislr.paths import VideoPathResolver, discover_video_root

    root = Path(dataset_root) if dataset_root else CISLR_DATA_DIR
    video_root = discover_video_root(str(root))
    if not video_root:
        return None, None
    return video_root, VideoPathResolver(video_root)


def normalize_label(label: str) -> str:
    """Normalise a raw dataset gloss to uppercase stripped form."""
    return label.strip().upper()


def normalize_landmarks(landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Translate landmarks so the wrist (index 0) is the origin,
    then scale by the distance from wrist (0) to middle-finger MCP (9).

    Returns a flat np.ndarray of shape (63,) = (21 × 3).
    Returns None if the hand has fewer than 21 landmarks.
    """
    if len(landmarks) < 21:
        return None

    arr = np.array(landmarks, dtype=np.float32)  # (21, 3)

    # Translate: subtract wrist position
    origin = arr[0].copy()
    arr -= origin

    # Scale: wrist-to-middle-MCP distance
    scale = np.linalg.norm(arr[9])
    if scale > 1e-6:
        arr /= scale

    return arr.flatten()  # (63,)


def build_window_feature(frame_vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Given a list of per-frame 63-d vectors, pad/trim to WINDOW_SIZE
    and concatenate into a single (315,) feature vector.
    Returns None if the list is empty.
    """
    if not frame_vectors:
        return None

    vectors = list(frame_vectors)

    # Pad with zeros if fewer frames than window
    while len(vectors) < WINDOW_SIZE:
        vectors.insert(0, np.zeros(FEATURES_PER_FRAME, dtype=np.float32))

    # Trim to the last WINDOW_SIZE frames
    vectors = vectors[-WINDOW_SIZE:]

    return np.concatenate(vectors)  # (315,)
