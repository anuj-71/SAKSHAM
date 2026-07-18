r"""
Build dataset manifest from extracted landmark NPZ files.

Scans dataset/landmarks/*.npz and cross-references against the curated
vocabulary to produce dataset/processed/manifest.csv.

Columns:
  uid, label, signer_id, video_path, landmark_path, frame_count, approved

Usage:
    .venv\Scripts\python.exe -m tools.dataset.build_manifest
"""
import csv
import logging
from pathlib import Path

import numpy as np

from tools.dataset.paths import LANDMARKS_DIR, PROCESSED_DIR, MANIFEST_PATH, ensure_dirs
from tools.dataset.vocabulary import get_label_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MANIFEST_FIELDS = [
    "uid", "label", "signer_id", "video_path",
    "landmark_path", "frame_count", "approved",
]


def main() -> int:
    ensure_dirs()
    label_map = get_label_map()
    known_labels = set(label_map.keys())

    npz_files = sorted(LANDMARKS_DIR.glob("*.npz"))
    if not npz_files:
        logging.warning("No NPZ files found in %s", LANDMARKS_DIR)
        return 0

    rows = []
    for npz_path in npz_files:
        try:
            with np.load(npz_path) as data:
                gloss = str(data.get("gloss", ""))
                signer_id = str(data.get("signer_id", ""))
                uid = str(data.get("uid", ""))
                video_path = str(data.get("video_path", ""))
                frame_count = int(data.get("frame_count", 0))
        except Exception as e:
            logging.warning("Failed to read %s: %s", npz_path.name, e)
            continue

        approved = gloss in known_labels and gloss != "unknown"

        rows.append({
            "uid": uid,
            "label": gloss,
            "signer_id": signer_id,
            "video_path": video_path,
            "landmark_path": str(npz_path),
            "frame_count": frame_count,
            "approved": str(approved),
        })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    approved_count = sum(1 for r in rows if r["approved"] == "True")
    logging.info("Wrote %s (%d rows, %d approved)", MANIFEST_PATH, len(rows), approved_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
