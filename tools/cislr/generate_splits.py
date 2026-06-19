r"""
Generate stratified train/val/test splits from vocabulary manifest.

Reads processed/vocabulary_manifest.csv and produces:
  - processed/train.csv
  - processed/val.csv
  - processed/test.csv

Split ratio: 70/15/15 stratified by gloss.

Usage:
    .\.venv\Scripts\python.exe -m tools.cislr.generate_splits
"""
import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from tools.cislr.utils import CISLR_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--processed-dir", type=str, default=None, help="Override processed directory")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir) if args.processed_dir else CISLR_DATA_DIR / "processed"
    rng = random.Random(args.seed)

    manifest_path = processed_dir / "vocabulary_manifest.csv"
    if not manifest_path.exists():
        logging.error("Manifest not found: %s", manifest_path)
        logging.error("Run build_vocabulary.py first.")
        return 1

    rows = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        logging.warning("Empty manifest; writing empty split CSVs")
        header = ["npz_path", "label", "label_index"]
        for name in ("train.csv", "val.csv", "test.csv"):
            p = processed_dir / name
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
        return 0

    gloss_groups = defaultdict(list)
    for r in rows:
        gloss_groups[r["label"]].append(r)

    train_rows = []
    val_rows = []
    test_rows = []

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

    header = ["npz_path", "label", "label_index"]
    splits = [("train.csv", train_rows), ("val.csv", val_rows), ("test.csv", test_rows)]
    for name, data in splits:
        path = processed_dir / name
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in data:
                w.writerow([r["npz_path"], r["label"], r["label_index"]])
        logging.info("Wrote %s (%d rows)", path, len(data))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
