import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from src.sign_language.training.loader import DatasetRecord
from src.sign_language.training.splits import split_records_by_signer
from tools.cislr.paths import (
    VideoPathResolver,
    detect_video_dir,
    discover_video_root,
    resolve_video_path,
    sum_mp4_bytes,
)
from tools.normalize_labels import TARGET_LABELS, detect_label_column, load_csv_rows, normalize_rows

SIGNER_COLUMN_CANDIDATES = (
    "signer_id",
    "signer",
    "person_id",
    "person",
    "speaker",
    "user",
)

VIDEO_COLUMN_CANDIDATES = (
    "video_path",
    "video",
    "video_id",
    "path",
    "filepath",
    "file",
    "filename",
    "uid",
    "id",
)

FLOAT32_BYTES = 4
LANDMARKS_PER_HAND = 21 * 3
BYTES_PER_FRAME = (
    1 * FLOAT32_BYTES +                       # timestamps
    2 * LANDMARKS_PER_HAND * FLOAT32_BYTES + # left/right landmarks
    4 * FLOAT32_BYTES +                      # left/right present + left/right confidence
    0
)


def detect_signer_column(fieldnames: Sequence[str]) -> Optional[str]:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in SIGNER_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def detect_video_column(
    fieldnames: Sequence[str],
    rows: Sequence[Dict[str, str]],
    video_dir: Optional[str],
    resolver: Optional[VideoPathResolver] = None,
) -> Optional[str]:
    lowered = {name.lower(): name for name in fieldnames}
    scored_columns: List[Tuple[int, str]] = []
    active_resolver = resolver
    if video_dir and active_resolver is None:
        active_resolver = VideoPathResolver(video_dir)

    for candidate in VIDEO_COLUMN_CANDIDATES:
        if candidate not in lowered:
            continue
        column_name = lowered[candidate]
        score = 0
        if active_resolver:
            for row in rows[:100]:
                if active_resolver.exists(row.get(column_name, "")):
                    score += 1
        scored_columns.append((score, column_name))

    if not scored_columns:
        return None

    scored_columns.sort(key=lambda item: (item[0], -fieldnames.index(item[1])), reverse=True)
    best_score, best_column = scored_columns[0]
    if best_score > 0:
        return best_column

    return best_column


def build_dummy_records(rows: Sequence[Dict[str, str]], signer_column: str) -> List[DatasetRecord]:
    records: List[DatasetRecord] = []
    for index, row in enumerate(rows):
        records.append(
            DatasetRecord(
                sample_id=f"CISLR_{index:06d}",
                label=row["saksham_label"],
                signer_id=row.get(signer_column, "").strip() or "UNKNOWN_SIGNER",
                frame_count=0,
                metadata=row,
                npz_path="",
                metadata_path="",
            )
        )
    return records


def get_video_frame_count(video_path: str) -> int:
    if cv2 is None:
        return 0
    capture = cv2.VideoCapture(video_path)
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return max(frame_count, 0)


def sum_directory_video_bytes(video_dir: Optional[str]) -> int:
    return sum_mp4_bytes(video_dir or "")


def audit_cislr_metadata(dataset_root: str, metadata_path: Optional[str] = None, label_column: Optional[str] = None) -> Dict:
    resolved_metadata_path = metadata_path or os.path.join(dataset_root, "dataset.csv")
    if not os.path.exists(resolved_metadata_path):
        raise FileNotFoundError(f"CISLR metadata CSV not found: {resolved_metadata_path}")

    fieldnames, rows = load_csv_rows(resolved_metadata_path)
    resolved_label_column = detect_label_column(fieldnames, preferred=label_column)
    normalized_rows = normalize_rows(rows, label_column=resolved_label_column)

    matched_rows = [row for row in normalized_rows if row.get("saksham_label")]
    video_dir = discover_video_root(dataset_root) or detect_video_dir(dataset_root)
    signer_column = detect_signer_column(fieldnames)
    video_resolver = VideoPathResolver(video_dir) if video_dir else None
    video_column = detect_video_column(
        fieldnames,
        normalized_rows,
        video_dir,
        resolver=video_resolver,
    )

    target_counts = {label: 0 for label in TARGET_LABELS}
    for row in matched_rows:
        target_counts[row["saksham_label"]] += 1

    filtered_raw_video_bytes = 0
    filtered_video_count = 0
    missing_video_count = 0
    projected_landmark_bytes = 0
    frame_count_samples = []

    if video_dir and video_column and video_resolver:
        seen_paths = set()
        for row in matched_rows:
            resolved_video_path = video_resolver.resolve(row.get(video_column, ""))
            if not resolved_video_path or resolved_video_path in seen_paths:
                continue
            seen_paths.add(resolved_video_path)
            if os.path.exists(resolved_video_path):
                filtered_video_count += 1
                filtered_raw_video_bytes += os.path.getsize(resolved_video_path)
                frame_count = get_video_frame_count(resolved_video_path)
                frame_count_samples.append(frame_count)
                projected_landmark_bytes += frame_count * BYTES_PER_FRAME
            else:
                missing_video_count += 1

    split_counts = {"train": 0, "val": 0, "test": 0}
    unique_signers = 0
    if signer_column:
        records = build_dummy_records(matched_rows, signer_column=signer_column)
        split_map = split_records_by_signer(records, seed=42)
        split_counts = {key: len(value) for key, value in split_map.items()}
        unique_signers = len({record.signer_id for record in records})

    total_raw_video_bytes = sum_directory_video_bytes(video_dir)
    metadata_bytes = os.path.getsize(resolved_metadata_path)
    recommended_working_bytes = filtered_raw_video_bytes + projected_landmark_bytes + metadata_bytes

    return {
        "dataset_root": dataset_root,
        "metadata_path": resolved_metadata_path,
        "metadata_bytes": metadata_bytes,
        "video_dir": video_dir or "",
        "label_column": resolved_label_column,
        "signer_column": signer_column or "",
        "video_column": video_column or "",
        "total_rows": len(normalized_rows),
        "matched_rows": len(matched_rows),
        "target_counts": target_counts,
        "unique_signers_in_filtered_set": unique_signers,
        "expected_split_sizes": split_counts,
        "total_raw_video_bytes": total_raw_video_bytes,
        "filtered_raw_video_bytes": filtered_raw_video_bytes,
        "filtered_video_count": filtered_video_count,
        "missing_filtered_videos": missing_video_count,
        "projected_landmark_tensor_bytes": projected_landmark_bytes,
        "recommended_working_bytes": recommended_working_bytes,
        "frame_count_summary": {
            "counted_videos": len(frame_count_samples),
            "min_frames": min(frame_count_samples) if frame_count_samples else 0,
            "max_frames": max(frame_count_samples) if frame_count_samples else 0,
            "avg_frames": (
                sum(frame_count_samples) / len(frame_count_samples)
                if frame_count_samples else 0.0
            ),
        },
    }


def human_bytes(size_bytes: int) -> str:
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    suffix_index = 0
    while size >= 1024.0 and suffix_index < len(suffixes) - 1:
        size /= 1024.0
        suffix_index += 1
    return f"{size:.2f} {suffixes[suffix_index]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CISLR metadata against the SAKSHAM target vocabulary.")
    parser.add_argument("--dataset-root", required=True, help="Root directory of the downloaded CISLR dataset.")
    parser.add_argument("--metadata-path", help="Optional explicit path to CISLR dataset.csv")
    parser.add_argument("--label-column", help="Optional explicit label/gloss column name.")
    parser.add_argument("--json-output", help="Optional path to save the audit report as JSON.")
    args = parser.parse_args()

    report = audit_cislr_metadata(
        dataset_root=args.dataset_root,
        metadata_path=args.metadata_path,
        label_column=args.label_column,
    )

    print(f"Dataset root: {report['dataset_root']}")
    print(f"Metadata CSV: {report['metadata_path']}")
    print(f"Detected label column: {report['label_column']}")
    print(f"Detected signer column: {report['signer_column'] or '<none>'}")
    print(f"Detected video column: {report['video_column'] or '<none>'}")
    print(f"Total rows: {report['total_rows']}")
    print(f"Matched rows: {report['matched_rows']}")
    print("Exact target label counts:")
    for label in TARGET_LABELS:
        print(f"- {label}: {report['target_counts'][label]}")

    print(f"Filtered unique signers: {report['unique_signers_in_filtered_set']}")
    print("Expected split sizes:")
    for split_name in ("train", "val", "test"):
        print(f"- {split_name}: {report['expected_split_sizes'][split_name]}")

    print("Storage:")
    print(f"- Metadata: {human_bytes(report['metadata_bytes'])}")
    print(f"- All raw CISLR videos on disk: {human_bytes(report['total_raw_video_bytes'])}")
    print(f"- Filtered raw videos on disk: {human_bytes(report['filtered_raw_video_bytes'])}")
    print(f"- Projected landmark tensors (uncompressed): {human_bytes(report['projected_landmark_tensor_bytes'])}")
    print(f"- Recommended working space: {human_bytes(report['recommended_working_bytes'])}")
    print(f"- Filtered videos found: {report['filtered_video_count']}")
    print(f"- Filtered videos missing: {report['missing_filtered_videos']}")

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as json_file:
            json.dump(report, json_file, indent=2)
        print(f"JSON report: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
