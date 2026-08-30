"""
tools/extract_asl_subset.py
===========================
Extracts ONLY the 14 target sign classes from the 37.4GB asl-signs.zip file.
Saves disk space by extracting only ~350MB of required parquet files instead of 37GB.

Usage:
  .venv\\Scripts\\python.exe tools/extract_asl_subset.py --zip_path "C:\\Users\\Dell\\Downloads\\asl-signs.zip" --dest asl_data
"""

import argparse
import csv
import io
import logging
import sys
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SelectiveZipExtractor")

TARGET_CLASSES = {
    "hello", "bye", "yes", "no", "please", "thankyou",
    "water", "drink", "food", "hungry", "home",
    "sick", "where", "sleep",
}


def main():
    parser = argparse.ArgumentParser(description="Selectively extract 14 target classes from asl-signs.zip")
    parser.add_argument("--zip_path", type=str, required=True, help="Path to downloaded asl-signs.zip")
    parser.add_argument("--dest", type=str, default="asl_data", help="Output destination folder (default: asl_data)")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    dest = Path(args.dest)

    if not zip_path.exists():
        log.error("Zip file not found at: %s", zip_path)
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    log.info("Opening zip archive: %s (%.2f GB)", zip_path.name, zip_path.stat().st_size / (1024 ** 3))

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = set(zf.namelist())

        # 1. Extract train.csv
        csv_candidates = [name for name in namelist if name.endswith("train.csv")]
        if not csv_candidates:
            log.error("train.csv not found inside zip archive!")
            sys.exit(1)

        csv_name = csv_candidates[0]
        log.info("Reading %s from zip...", csv_name)
        with zf.open(csv_name) as f:
            csv_bytes = f.read()

        # Save train.csv to destination
        out_csv = dest / "train.csv"
        with open(out_csv, "wb") as f:
            f.write(csv_bytes)
        log.info("Saved %s to destination.", out_csv)

        # 2. Filter target classes
        csv_text = io.StringIO(csv_bytes.decode("utf-8"))
        reader = csv.DictReader(csv_text)

        target_paths = []
        class_counts = {c: 0 for c in TARGET_CLASSES}

        for row in reader:
            sign = str(row.get("sign", "")).strip().lower()
            if sign in TARGET_CLASSES:
                rel_path = str(row.get("path", "")).replace("\\", "/")
                target_paths.append((rel_path, sign))
                class_counts[sign] += 1

        log.info("Found %d matching sequences across %d classes.", len(target_paths), len(TARGET_CLASSES))
        for sign in sorted(TARGET_CLASSES):
            log.info("  %-12s: %4d sequences", sign, class_counts[sign])

        # 3. Selectively extract only matching parquet files
        log.info("Extracting %d parquet files into %s...", len(target_paths), dest)
        extracted_count = 0

        for idx, (rel_path, sign) in enumerate(target_paths, 1):
            # Check path variants in zip
            possible_zip_names = [rel_path, f"train_landmark_files/{rel_path}", f"./{rel_path}"]
            found_zip_name = None
            for p in possible_zip_names:
                if p in namelist:
                    found_zip_name = p
                    break

            if found_zip_name:
                out_file = dest / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(found_zip_name) as src, open(out_file, "wb") as dst:
                    dst.write(src.read())
                extracted_count += 1

            if idx % 500 == 0 or idx == len(target_paths):
                log.info("Extracted %4d / %4d files...", idx, len(target_paths))

        log.info("=" * 60)
        log.info("EXTRACTION COMPLETE! Total files extracted: %d", extracted_count)
        log.info("Destination directory ready: %s", dest.resolve())
        log.info("You can now run: .venv\\Scripts\\python.exe tools/import_google_asl.py --source_dir %s", dest)
        log.info("=" * 60)


if __name__ == "__main__":
    main()
