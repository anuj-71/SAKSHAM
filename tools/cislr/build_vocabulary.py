r"""
Build training vocabulary from extracted CISLR NPZ files.

Scans accepted NPZ files, computes gloss frequency,
filters to glosses with >= min-samples, and writes:
  - processed/label_map.json   ({gloss: index, ...})
  - processed/vocabulary_manifest.csv  (npz_path,label,label_index)

Usage:
    .venv\Scripts\python.exe -m tools.cislr.build_vocabulary
"""
import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from tools.cislr.utils import LANDMARKS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum accepted samples per gloss")
    parser.add_argument("--accepted-dir", type=str, default=None, help="Override accepted NPZ directory")
    args = parser.parse_args()

    accepted_dir = Path(args.accepted_dir) if args.accepted_dir else Path(LANDMARKS_DIR) / "accepted"
    processed_dir = accepted_dir.parents[1] / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(accepted_dir.glob("*.npz"))
    if not npz_files:
        logging.warning("No NPZ files found in %s", accepted_dir)
        # Write empty outputs
        label_map_path = processed_dir / "label_map.json"
        manifest_path = processed_dir / "vocabulary_manifest.csv"
        with open(label_map_path, "w") as f:
            json.dump({}, f)
        with open(manifest_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["npz_path", "label", "label_index"])
        logging.info("Wrote empty label_map and manifest (no data)")
        return 0

    freq = Counter()
    file_glosses = []

    for npz_path in npz_files:
        with np.load(npz_path) as data:
            gloss = str(data["gloss"])
        freq[gloss] += 1
        file_glosses.append((str(npz_path), gloss))

    logging.info("Total NPZ files: %d", len(npz_files))
    logging.info("Unique glosses: %d", len(freq))

    candidate_glosses = {g for g, c in freq.items() if c >= args.min_samples}
    logging.info("Glosses with >=%d samples: %d", args.min_samples, len(candidate_glosses))

    filtered = [(p, g) for p, g in file_glosses if g in candidate_glosses]
    sorted_glosses = sorted(candidate_glosses)
    label_map = {g: i for i, g in enumerate(sorted_glosses)}

    label_map_path = processed_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logging.info("Wrote %s (%d classes)", label_map_path, len(label_map))

    manifest_path = processed_dir / "vocabulary_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["npz_path", "label", "label_index"])
        for npz_path, gloss in filtered:
            w.writerow([npz_path, gloss, label_map[gloss]])
    logging.info("Wrote %s (%d rows)", manifest_path, len(filtered))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
