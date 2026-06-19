"""
CISLR video path discovery utilities.

Handles common layouts such as:
  cislr_data/CISLR_v1.5-a_videos/*.mp4
  cislr_data/CISLR_v1.5-a_videos/CISLR_v1.5-a_videos/*.mp4
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence

VIDEO_DIR_CANDIDATES = (
    "CISLR_v1.5-a_videos",
    "raw_videos",
    "videos",
    "CISLR_videos",
    ".",
)


def count_mp4_files(directory: str) -> int:
    """Count MP4 files recursively under a directory."""
    if not directory or not os.path.isdir(directory):
        return 0
    return sum(1 for _ in Path(directory).rglob("*.mp4"))


def sum_mp4_bytes(directory: str) -> int:
    """Sum byte size of all MP4 files recursively under a directory."""
    if not directory or not os.path.isdir(directory):
        return 0
    total = 0
    for path in Path(directory).rglob("*.mp4"):
        total += path.stat().st_size
    return total


def discover_video_root(dataset_root: str) -> Optional[str]:
    """
    Locate the directory containing CISLR MP4 files.

    Scans known candidate folders and nested subdirectories, returning the
    path with the largest MP4 count.
    """
    best_path: Optional[str] = None
    best_count = 0

    for candidate in VIDEO_DIR_CANDIDATES:
        base = os.path.join(dataset_root, candidate)
        if not os.path.isdir(base):
            continue

        for root, _, files in os.walk(base):
            mp4_count = sum(1 for name in files if name.lower().endswith(".mp4"))
            if mp4_count > best_count:
                best_count = mp4_count
                best_path = root

    return best_path


def detect_video_dir(dataset_root: str) -> Optional[str]:
    """Return the best MP4 root directory under dataset_root."""
    return discover_video_root(dataset_root)


class VideoPathResolver:
    """Resolve metadata video IDs to on-disk MP4 paths using a basename index."""

    def __init__(self, video_root: str):
        self.video_root = video_root
        self._index: Dict[str, str] = {}
        self._build_index()

    @property
    def indexed_video_count(self) -> int:
        return len(self._index)

    def _build_index(self) -> None:
        root = Path(self.video_root)
        if not root.is_dir():
            return
        for path in root.rglob("*.mp4"):
            self._index[path.name.lower()] = str(path)

    def resolve(self, raw_value: str) -> str:
        token = str(raw_value or "").strip()
        if not token:
            return ""

        basename = os.path.basename(token)
        if os.path.splitext(basename)[1]:
            candidates = [basename]
        else:
            candidates = [f"{basename}.mp4", basename]

        for candidate in candidates:
            indexed = self._index.get(candidate.lower())
            if indexed:
                return indexed
            direct = os.path.join(self.video_root, candidate)
            if os.path.exists(direct):
                return direct

        return os.path.join(self.video_root, candidates[0])

    def exists(self, raw_value: str) -> bool:
        resolved = self.resolve(raw_value)
        return bool(resolved) and os.path.exists(resolved)


def resolve_video_path(
    video_dir: str,
    raw_value: str,
    resolver: Optional[VideoPathResolver] = None,
) -> str:
    """Resolve a metadata video token to a filesystem path."""
    if resolver is not None:
        return resolver.resolve(raw_value)
    if not video_dir:
        return ""
    return VideoPathResolver(video_dir).resolve(raw_value)
