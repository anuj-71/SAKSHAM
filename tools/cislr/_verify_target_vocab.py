"""Verification of candidate target-vocabulary samples from the CISLR dataset.

For each candidate UID this script:
  * Confirms the .mp4 exists.
  * Opens the file with OpenCV and reads every frame to detect truncation/corruption.
  * Reports frame count, fps, resolution and duration.
  * Computes an integrity flag (OK / READ_ERROR / EMPTY / MISSING).

No files are modified or copied. Read-only inspection.
"""

from __future__ import annotations

import csv
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_CSV = os.path.join(ROOT, "cislr_data", "dataset.csv")
VIDEO_DIR = os.path.join(
    ROOT, "cislr_data", "CISLR_v1.5-a_videos", "CISLR_v1.5-a_videos"
)

TARGETS = [
    "thank you",
    "yes",
    "no",
    "help",
    "water",
    "eat",
    "drink",
    "stop",
    "please",
    "hello",
]


def load_index() -> dict[str, list[tuple[str, str, str]]]:
    """Return {normalized_gloss: [(uid, duration, category), ...]}."""
    index: dict[str, list[tuple[str, str, str]]] = {}
    with open(DATASET_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gloss = (row.get("gloss") or "").strip()
            if not gloss:
                continue
            index.setdefault(gloss.lower(), []).append(
                (row["uid"].strip(), row.get("duration", "").strip(), row.get("category", "").strip())
            )
    return index


def probe_video(uid: str) -> dict:
    path = os.path.join(VIDEO_DIR, f"{uid}.mp4")
    info = {"uid": uid, "path": path, "exists": os.path.exists(path)}
    if not info["exists"]:
        info["status"] = "MISSING"
        info["size_bytes"] = None
        return info
    info["size_bytes"] = os.path.getsize(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        info["status"] = "READ_ERROR"
        return info
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    read_count = 0
    last_ok = False
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        read_count += 1
        last_ok = True
    cap.release()
    info.update(
        {
            "declared_frames": frames,
            "read_frames": read_count,
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "duration_sec": round(read_count / fps, 3) if fps else None,
        }
    )
    if read_count == 0:
        info["status"] = "EMPTY"
    elif frames and read_count < frames:
        info["status"] = "TRUNCATED"
    else:
        info["status"] = "OK"
    return info


def main() -> int:
    index = load_index()
    print(f"Loaded {sum(len(v) for v in index.values())} rows across "
          f"{len(index)} unique glosses.\n")

    for target in TARGETS:
        rows = index.get(target, [])
        print("=" * 70)
        print(f"TARGET: {target!r}   candidates_in_csv: {len(rows)}")
        if not rows:
            print("  -> NO SAMPLES in dataset for this gloss.\n")
            continue
        for uid, dur, cat in rows:
            info = probe_video(uid)
            print(
                f"  uid={uid:<20} status={info.get('status'):<10} "
                f"size={info.get('size_bytes')}B  read_frames={info.get('read_frames')}/"
                f"{info.get('declared_frames')}  {info.get('width')}x{info.get('height')}@"
                f"{info.get('fps')}fps  duration={info.get('duration_sec')}s  "
                f"csv_dur={dur}  csv_cat={cat}"
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
