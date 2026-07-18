r"""
Generate the label_map.json for the curated dataset.

Writes to dataset/processed/label_map.json with format {label: index}.
Index 0 is reserved for "unknown" (non-sign gestures).

Usage:
    .venv\Scripts\python.exe -m tools.dataset.generate_label_map
"""
import json
import logging

from tools.dataset.paths import PROCESSED_DIR, LABEL_MAP_PATH, ensure_dirs
from tools.dataset.vocabulary import get_label_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ensure_dirs()
    label_map = get_label_map()
    num_classes = len(label_map)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    logging.info("Wrote %s (%d classes, unknown=index 0)", LABEL_MAP_PATH, num_classes)
    logging.info("Vocabulary: %s", list(label_map.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
