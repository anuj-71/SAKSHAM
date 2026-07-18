r"""
Generate stratified train/val/test splits from the dataset manifest.

Reads dataset/processed/manifest.csv (approved rows only) and produces:
  - dataset/processed/train.csv
  - dataset/processed/val.csv
  - dataset/processed/test.csv

Split ratio: 70/15/15 stratified by label.

Usage:
    .venv\Scripts\python.exe -m tools.dataset.generate_splits
"""
import argparse
import csv
import logging
import random
from collections import defaultdict
from pathlib import Path

from tools.dataset.paths import PROCESSED_DIR, MANIFEST_PATH, TRAIN_CSV, VAL_CSV, TEST_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SPLIT_FIELDS = ["npz_path", "label", "label_index"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    if not MANIFEST_PATH.exists():
        logging.error("Manifest not found: %s", MANIFEST_PATH)
        logging.error("Run build_manifest.py first.")
        return 1

    rows = []
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("approved", "").lower() == "true":
                rows.append(r)

    if not rows:
        logging.warning("No approved rows in manifest; writing empty split CSVs")
        for path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(SPLIT_FIELDS)
        return 0

    # Build label index mapping from manifest labels (string-based, sorted)
    unique_labels = sorted({r["label"] for r in rows})
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    gloss_groups = defaultdict(list)
    for r in rows:
        gloss_groups[r["label"]].append(r)

    train_rows, val_rows, test_rows = [], [], []

    for gloss, group in gloss_groups.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * 0.70))
        n_val = max(0, round(n * 0.15))
        if n_val == 0 and n - n_train > 0:
            n_val = 1
        n_test = n - n_train - n_val
        if n_test <= 0 and n_val > 1:
            n_val -= 1
            n_test = 1
        while n_train + n_val + n_test > n:
            if n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                n_train -= 1

        train_rows.extend(group[:n_train])
        val_rows.extend(group[n_train:n_train + n_val])
        test_rows.extend(group[n_train + n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    def write_split(path: Path, data: list):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(SPLIT_FIELDS)
            for r in data:
                w.writerow([r["landmark_path"], r["label"], label_to_idx[r["label"]]])
        logging.info("Wrote %s (%d rows)", path, len(data))

    write_split(TRAIN_CSV, train_rows)
    write_split(VAL_CSV, val_rows)
    write_split(TEST_CSV, test_rows)

    total = len(train_rows) + len(val_rows) + len(test_rows)
    logging.info("Split: train=%d val=%d test=%d (total approved=%d)", len(train_rows), len(val_rows), len(test_rows), total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
