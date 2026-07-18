r"""
Batch MediaPipe Hands landmark extraction for the curated SAKSHAM dataset.

Scans dataset/raw_videos/{gloss}/{signer}/*.mp4 and writes NPZ files to
dataset/landmarks/{gloss}_{signer}_{video_uid}.npz in canonical format.

Usage:
    .venv\Scripts\python.exe -m tools.dataset.extract_landmarks
    .venv\Scripts\python.exe -m tools.dataset.extract_landmarks --gloss hello
"""
import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from tools.dataset.paths import RAW_VIDEOS_DIR, LANDMARKS_DIR, ensure_dirs
from src.sign_language.tracker import HandTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.strip(". ")
    return sanitized or "unnamed"


def extract_from_video(video_path: str, tracker: HandTracker) -> tuple:
    cap = cv2.VideoCapture(video_path)
    left_hand_list, right_hand_list = [], []
    left_present_list, right_present_list = [], []
    left_conf_list, right_conf_list = [], []
    missing_hand = np.zeros((21, 3), dtype=np.float32)

    if not cap.isOpened():
        logging.warning("Unable to open video: %s", video_path)
        return (None, None, None, None, None, None, None, False, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected, data = tracker.process_frame(frame)

        l_land = missing_hand
        r_land = missing_hand
        l_pres = 0.0
        r_pres = 0.0
        l_conf = 0.0
        r_conf = 0.0

        if detected:
            left = data.get("left_hand", {})
            right = data.get("right_hand", {})

            if left.get("present"):
                lm = left["landmarks"]
                if len(lm) >= 21:
                    l_land = np.array(lm[:21], dtype=np.float32)
                    l_pres = 1.0
                    l_conf = float(left.get("confidence", 0.0))

            if right.get("present"):
                lm = right["landmarks"]
                if len(lm) >= 21:
                    r_land = np.array(lm[:21], dtype=np.float32)
                    r_pres = 1.0
                    r_conf = float(right.get("confidence", 0.0))

        left_hand_list.append(l_land)
        right_hand_list.append(r_land)
        left_present_list.append(l_pres)
        right_present_list.append(r_pres)
        left_conf_list.append(l_conf)
        right_conf_list.append(r_conf)

    cap.release()
    frame_count = len(left_hand_list)
    if frame_count == 0:
        return (None, None, None, None, None, None, None, True, 0)

    return (
        np.stack(left_hand_list, axis=0).astype(np.float32),
        np.stack(right_hand_list, axis=0).astype(np.float32),
        np.array(left_present_list, dtype=np.float32),
        np.array(right_present_list, dtype=np.float32),
        np.array(left_conf_list, dtype=np.float32),
        np.array(right_conf_list, dtype=np.float32),
        np.arange(frame_count, dtype=np.float32),
        True,
        frame_count,
    )


def discover_videos(raw_root: Path, target_gloss: Optional[str] = None) -> list[dict]:
    videos = []
    gloss_dirs = sorted(raw_root.iterdir()) if raw_root.is_dir() else []

    for gloss_dir in gloss_dirs:
        if not gloss_dir.is_dir():
            continue
        gloss = gloss_dir.name
        if target_gloss and gloss != target_gloss:
            continue

        for signer_dir in sorted(gloss_dir.iterdir()):
            if not signer_dir.is_dir():
                continue
            signer = signer_dir.name
            for mp4_path in sorted(signer_dir.glob("*.mp4")):
                videos.append({
                    "path": str(mp4_path),
                    "gloss": gloss,
                    "signer": signer,
                    "uid": mp4_path.stem,
                })
    return videos


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gloss", type=str, default=None, help="Extract only this gloss")
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process (0 = all)")
    args = parser.parse_args()

    ensure_dirs()
    videos = discover_videos(RAW_VIDEOS_DIR, target_gloss=args.gloss)

    if not videos:
        logging.warning("No MP4 files found in %s", RAW_VIDEOS_DIR)
        return 0

    logging.info("Found %d videos to process", len(videos))

    try:
        tracker = HandTracker()
    except Exception as e:
        logging.error("Failed to initialize HandTracker: %s", e)
        return 3

    skipped = 0
    success = 0
    failed = 0
    limit = args.limit or len(videos)

    for idx, info in enumerate(videos):
        if idx >= limit:
            break

        gloss = info["gloss"]
        signer = info["signer"]
        uid = info["uid"]
        video_path = info["path"]

        out_name = f"{sanitize_filename(gloss)}_{sanitize_filename(signer)}_{sanitize_filename(uid)}.npz"
        out_path = LANDMARKS_DIR / out_name

        if out_path.exists():
            skipped += 1
            continue

        result = extract_from_video(video_path, tracker)
        left_hand, right_hand, left_present, right_present, left_conf, right_conf, timestamps, opened, frame_count = result

        if not opened:
            failed += 1
            continue

        if np.max(left_present) == 0.0 and np.max(right_present) == 0.0:
            failed += 1
            continue

        np.savez_compressed(
            out_path,
            left_hand=left_hand.astype(np.float32),
            right_hand=right_hand.astype(np.float32),
            left_present=left_present.astype(np.float32),
            right_present=right_present.astype(np.float32),
            left_confidence=left_conf.astype(np.float32),
            right_confidence=right_conf.astype(np.float32),
            timestamps=timestamps.astype(np.float32),
            gloss=gloss,
            signer_id=signer,
            uid=uid,
            video_path=video_path,
            frame_count=int(frame_count),
        )

        success += 1

    logging.info("Extraction complete: success=%d failed=%d skipped=%d", success, failed, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
