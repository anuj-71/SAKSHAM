"""
tools/download_asl_subset.py

Downloads only the sequences needed for SAKSHAM's vocabulary from the
Kaggle `asl-signs` competition, instead of pulling the full ~35GB archive.

Strategy:
  1. Download train.csv only (small) via `kaggle competitions download -f`.
  2. Filter rows to the target signs.
  3. Download each required train_landmark_files/<participant>/<sequence>.parquet
     individually via the same -f flag, into asl_data/, with rate limiting
     and retry/backoff (Kaggle throttles bursts of single-file API calls).

Requires: `kaggle` CLI installed and authenticated, and that you've accepted
the asl-signs competition rules on kaggle.com first.

Usage:
    python tools/download_asl_subset.py --dest asl_data --workers 2 --delay 0.5
"""
import argparse
import csv
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

COMPETITION = "asl-signs"

TARGET_CLASSES = [
    "hello", "bye", "yes", "no", "please", "thankyou",
    "water", "drink", "food", "hungry", "help", "home",
    "sick", "where", "sleep",
]

MAX_RETRIES = 4
BASE_BACKOFF = 3.0  # seconds, doubles each retry


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def download_train_csv(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "train.csv"
    if csv_path.exists():
        print("[1/3] train.csv already present, skipping download.")
        return csv_path

    print("[1/3] Downloading train.csv ...")
    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            run([
                "kaggle", "competitions", "download", "-c", COMPETITION,
                "-f", "train.csv", "-p", str(dest),
            ])
            zip_path = dest / "train.csv.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(dest)
                zip_path.unlink()
            if csv_path.exists():
                return csv_path
            last_err = "download reported success but train.csv missing"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            if attempt < MAX_RETRIES:
                backoff = BASE_BACKOFF * (2 ** (attempt - 1))
                if "429" in last_err or "rate" in last_err.lower():
                    backoff *= 2
                print(f"  attempt {attempt} failed ({last_err}), retrying in {backoff:.0f}s ...")
                time.sleep(backoff)
    raise FileNotFoundError(f"train.csv did not download after {MAX_RETRIES} attempts: {last_err}")


def filter_rows(csv_path: Path) -> list[dict]:
    print("[2/3] Filtering to target classes ...")
    wanted = set(TARGET_CLASSES)
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["sign"] in wanted:
                rows.append(row)
    found_classes = {r["sign"] for r in rows}
    missing = wanted - found_classes
    if missing:
        print(f"  WARNING: these classes were not found in train.csv: {sorted(missing)}")
    print(f"  {len(rows)} sequences across {len(found_classes)} classes matched.")
    return rows


def download_one(path: str, dest: Path, delay: float) -> tuple[str, bool, str]:
    out_dir = dest / str(Path(path).parent)
    out_file = dest / path
    if out_file.exists():
        return path, True, "already present"
    out_dir.mkdir(parents=True, exist_ok=True)

    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(delay)  # throttle every attempt, not just the first
            run([
                "kaggle", "competitions", "download", "-c", COMPETITION,
                "-f", path, "-p", str(out_dir),
            ])
            zipped = out_dir / (Path(path).name + ".zip")
            if zipped.exists():
                with zipfile.ZipFile(zipped) as zf:
                    zf.extractall(out_dir)
                zipped.unlink()
            if out_file.exists():
                return path, True, "ok"
            last_err = "download reported success but file missing"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            is_rate_limit = "429" in last_err or "rate" in last_err.lower()
            if attempt < MAX_RETRIES:
                backoff = BASE_BACKOFF * (2 ** (attempt - 1))
                if is_rate_limit:
                    backoff *= 2  # back off harder on explicit rate-limit signals
                time.sleep(backoff)
                continue
    return path, False, last_err


def download_landmark_files(rows: list[dict], dest: Path, workers: int, delay: float) -> None:
    print(f"[3/3] Downloading {len(rows)} landmark files "
          f"({workers} workers, {delay}s throttle/request) ...")
    paths = [row["path"] for row in rows]
    ok, failed = 0, []
    counter_lock = Lock()
    processed = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, p, dest, delay): p for p in paths}
        for fut in as_completed(futures):
            path, success, msg = fut.result()
            with counter_lock:
                processed += 1
                if success:
                    ok += 1
                else:
                    failed.append((path, msg))
                if processed % 50 == 0 or processed == len(paths):
                    print(f"  {processed}/{len(paths)} processed, {ok} ok, {len(failed)} failed")

    if failed:
        fail_log = dest / "download_failures.txt"
        with open(fail_log, "w") as f:
            for path, msg in failed:
                f.write(f"{path}\t{msg}\n")
        print(f"  {len(failed)} files failed after retries — see {fail_log}")
        print("  Rerun this same command to retry just the failed/missing ones "
              "(already-downloaded files are skipped).")


def write_manifest(rows: list[dict], dest: Path) -> None:
    manifest_path = dest / "subset_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "participant_id", "sequence_id", "sign"])
        for row in rows:
            writer.writerow([row["path"], row["participant_id"], row["sequence_id"], row["sign"]])
    print(f"Manifest written to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="asl_data", help="Output directory")
    parser.add_argument("--workers", type=int, default=2,
                         help="Parallel download workers (keep low, Kaggle rate-limits)")
    parser.add_argument("--delay", type=float, default=0.5,
                         help="Seconds to wait before each request, per worker")
    args = parser.parse_args()

    dest = Path(args.dest)
    csv_path = download_train_csv(dest)
    rows = filter_rows(csv_path)
    if not rows:
        print("No matching rows found — check TARGET_CLASSES against train.csv sign names.")
        sys.exit(1)
    download_landmark_files(rows, dest, args.workers, args.delay)
    write_manifest(rows, dest)
    print("Done. Point tools/import_google_asl.py --source_dir at this directory.")


if __name__ == "__main__":
    main()
