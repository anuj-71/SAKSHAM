"""
Download the CISLR dataset from Hugging Face into cislr_data/.

Prerequisites:
  1. Accept the dataset terms at https://huggingface.co/datasets/Exploration-Lab/CISLR
  2. Authenticate locally:
       hf auth login
     or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN in the environment.

Usage:
  python tools/cislr/download_dataset.py
  python tools/cislr/download_dataset.py --metadata-only
  python tools/cislr/download_dataset.py --skip-videos
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.cislr.utils import CISLR_DATA_DIR, ensure_dirs
from tools.cislr.paths import count_mp4_files, discover_video_root

REPO_ID = "Exploration-Lab/CISLR"
METADATA_FILES = ("dataset.csv", "prototype.csv", "test.csv", "README.md")
VIDEO_ZIP_HF_PATH = "CISLR_v1.5-a_videos/CISLR_v1.5-a_videos.zip"
VIDEO_DIR_NAME = "CISLR_v1.5-a_videos"


def _import_hf():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc
    return hf_hub_download


def download_metadata(local_dir: Path) -> list[Path]:
    hf_hub_download = _import_hf()
    downloaded: list[Path] = []
    for filename in METADATA_FILES:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_dir),
        )
        downloaded.append(Path(path))
        print(f"Downloaded metadata: {path}")
    return downloaded


def download_video_archive(local_dir: Path) -> Path:
    hf_hub_download = _import_hf()
    archive_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=VIDEO_ZIP_HF_PATH,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    print(f"Downloaded video archive: {archive_path}")
    return Path(archive_path)


def extract_video_archive(archive_path: Path, output_dir: Path) -> Path:
    video_root = output_dir / VIDEO_DIR_NAME
    video_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        members = archive.namelist()
        print(f"Extracting {len(members)} entries from {archive_path.name} ...")
        archive.extractall(video_root)

    mp4_files = list(video_root.rglob("*.mp4"))
    if not mp4_files and (video_root / VIDEO_DIR_NAME).exists():
        nested = video_root / VIDEO_DIR_NAME
        for item in nested.iterdir():
            target = video_root / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        shutil.rmtree(nested, ignore_errors=True)
        mp4_files = list(video_root.rglob("*.mp4"))

    print(f"Extracted {len(mp4_files)} MP4 files to {video_root}")
    return video_root


def verify_layout(dataset_root: Path) -> dict:
    dataset_csv = dataset_root / "dataset.csv"
    video_dir = discover_video_root(str(dataset_root))
    return {
        "dataset_root": str(dataset_root),
        "dataset_csv_exists": dataset_csv.exists(),
        "prototype_csv_exists": (dataset_root / "prototype.csv").exists(),
        "test_csv_exists": (dataset_root / "test.csv").exists(),
        "video_dir_exists": bool(video_dir),
        "video_search_root": video_dir or "",
        "video_count": count_mp4_files(video_dir) if video_dir else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CISLR into cislr_data/.")
    parser.add_argument(
        "--output-dir",
        default=str(CISLR_DATA_DIR),
        help="Destination directory (default: cislr_data/)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download CSV metadata only (no videos).",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Alias for --metadata-only.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download the video zip but do not extract it.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dirs()

    try:
        download_metadata(output_dir)
        metadata_only = args.metadata_only or args.skip_videos
        if not metadata_only:
            archive_path = download_video_archive(output_dir)
            if not args.skip_extract:
                extract_video_archive(archive_path, output_dir)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "\nCISLR is a gated dataset. Ensure you have accepted the terms and run:\n"
            "  hf auth login\n"
            "or set HF_TOKEN before retrying.",
            file=sys.stderr,
        )
        return 1

    layout = verify_layout(output_dir)
    print("\nLayout verification:")
    for key, value in layout.items():
        print(f"  {key}: {value}")

    if not layout["dataset_csv_exists"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
