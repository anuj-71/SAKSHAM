"""
Full CISLR dataset audit for SAKSHAM Phase 4C integration feasibility.

Produces JSON + Markdown reports covering:
  - Total samples and unique gloss count
  - Class distribution and top-N frequent signs
  - Signer metadata (if present)
  - Video format / storage layout
  - Pipeline suitability assessment (MediaPipe → GRU → ONNX → SAKSHAM)

Usage:
  python tools/cislr/audit_dataset.py --dataset-root cislr_data
  python tools/cislr/audit_dataset.py --dataset-root cislr_data --video-sample-size 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_cislr_metadata import (
    detect_signer_column,
    detect_video_column,
    get_video_frame_count,
    human_bytes,
    sum_directory_video_bytes,
)
from tools.cislr.paths import VideoPathResolver, count_mp4_files, discover_video_root
from tools.cislr.utils import REPORTS_DIR, ensure_dirs
from tools.normalize_labels import (
    TARGET_LABELS,
    canonicalize_label,
    detect_label_column,
    load_csv_rows,
    map_to_saksham_label,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

VIDEO_DIR_CANDIDATES = (
    "CISLR_v1.5-a_videos",
    "raw_videos",
    "videos",
    "CISLR_videos",
)

V1_VOCABULARY_TIERS = (
    {"name": "tier_bootstrap", "min_samples_per_gloss": 1, "recommended_v1_size": 10},
    {"name": "tier_v1", "min_samples_per_gloss": 2, "recommended_v1_size": 50},
    {"name": "tier_v1_expanded", "min_samples_per_gloss": 5, "recommended_v1_size": 150},
    {"name": "tier_v2", "min_samples_per_gloss": 10, "recommended_v1_size": 500},
)

FLOAT32_BYTES = 4
LANDMARKS_PER_HAND = 21 * 3
BYTES_PER_FRAME = (
    FLOAT32_BYTES
    + 2 * LANDMARKS_PER_HAND * FLOAT32_BYTES
    + 4 * FLOAT32_BYTES
)


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile / 100.0) * (len(ordered) - 1)))
    return float(ordered[index])


def _distribution_summary(counts: Counter) -> Dict[str, Any]:
    if not counts:
        return {
            "classes": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "singleton_classes": 0,
            "classes_with_2_or_more": 0,
            "classes_with_5_or_more": 0,
            "classes_with_10_or_more": 0,
        }

    per_class = list(counts.values())
    return {
        "classes": len(counts),
        "min": min(per_class),
        "max": max(per_class),
        "mean": sum(per_class) / len(per_class),
        "median": _percentile(per_class, 50),
        "p90": _percentile(per_class, 90),
        "singleton_classes": sum(1 for value in per_class if value == 1),
        "classes_with_2_or_more": sum(1 for value in per_class if value >= 2),
        "classes_with_5_or_more": sum(1 for value in per_class if value >= 5),
        "classes_with_10_or_more": sum(1 for value in per_class if value >= 10),
    }


def _inspect_video_file(video_path: str) -> Dict[str, Any]:
    if cv2 is None or not os.path.exists(video_path):
        return {
            "path": video_path,
            "exists": os.path.exists(video_path),
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "frame_count": 0,
            "duration_sec": 0.0,
            "fourcc": "",
            "file_bytes": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
        }

    capture = cv2.VideoCapture(video_path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = "".join(chr((fourcc_int >> 8 * index) & 0xFF) for index in range(4)).strip()
    finally:
        capture.release()

    duration_sec = (frame_count / fps) if fps > 0 else 0.0
    return {
        "path": video_path,
        "exists": True,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_sec": round(duration_sec, 3),
        "fourcc": fourcc,
        "file_bytes": os.path.getsize(video_path),
    }


def _sample_video_paths(
    rows: Sequence[Dict[str, str]],
    video_column: Optional[str],
    resolver: Optional[VideoPathResolver],
    sample_size: int,
) -> List[str]:
    if not video_column or resolver is None:
        return []

    seen = set()
    sampled: List[str] = []
    for row in rows:
        resolved = resolver.resolve(row.get(video_column, ""))
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        if os.path.exists(resolved):
            sampled.append(resolved)
        if len(sampled) >= sample_size:
            break
    return sampled


def _assess_pipeline_suitability(
    gloss_distribution: Dict[str, Any],
    signer_summary: Dict[str, Any],
    video_summary: Dict[str, Any],
    saksham_overlap: Dict[str, int],
) -> Dict[str, Any]:
    singleton_ratio = 0.0
    if gloss_distribution["classes"]:
        singleton_ratio = gloss_distribution["singleton_classes"] / gloss_distribution["classes"]

    risks = []
    if gloss_distribution["classes_with_10_or_more"] < 50:
        risks.append(
            "Most glosses have fewer than 10 videos; standard supervised multi-class GRU "
            "training will be severely data-starved without filtering or augmentation."
        )
    if singleton_ratio > 0.5:
        risks.append(
            f"{singleton_ratio:.0%} of gloss classes appear to be singletons (1 video). "
            "Prototype/one-shot or transfer learning is more realistic than plain GRU softmax."
        )
    if signer_summary.get("unique_signers", 0) < 5:
        risks.append("Limited signer diversity increases overfitting risk for signer-based splits.")

    strengths = [
        "Word-level RGB videos are compatible with offline MediaPipe Hands extraction.",
        "SAKSHAM already stores accepted clips as NPZ landmark tensors with left/right hands.",
        "Existing normalization and GRU preprocessing modules align with CISLR landmark conversion.",
        "ONNX export path is feasible once a compact GRU classifier is trained on fixed-length tensors.",
    ]

    if video_summary.get("sampled_videos", 0) > 0:
        avg_duration = video_summary.get("avg_duration_sec", 0.0)
        if avg_duration and avg_duration < 8.0:
            strengths.append(
                f"Short average clip duration (~{avg_duration:.1f}s) suits fixed window GRU inference."
            )

    saksham_matched = sum(1 for count in saksham_overlap.values() if count > 0)
    if saksham_matched:
        strengths.append(
            f"{saksham_matched} SAKSHAM V1 target glosses have direct CISLR rows for bootstrap validation."
        )

    return {
        "mediapipe_landmarks": {
            "suitable": True,
            "confidence": "high",
            "notes": "Dictionary-style cropped signer videos are a strong fit for 2D hand landmarks.",
        },
        "sequence_windows": {
            "suitable": True,
            "confidence": "high",
            "notes": "Use 24-frame padded sequences to match existing SAKSHAM preprocessing.",
        },
        "gru_training": {
            "suitable": gloss_distribution["classes_with_5_or_more"] >= 100,
            "confidence": "medium" if gloss_distribution["classes_with_5_or_more"] >= 100 else "low",
            "notes": (
                "Full 4.7k vocabulary is not realistic for vanilla supervised GRU. "
                "Filter to high-count glosses or use transfer/augmentation."
            ),
        },
        "onnx_export": {
            "suitable": True,
            "confidence": "high",
            "notes": "GRU with fixed input [batch, 24, 128] exports cleanly to ONNX Runtime.",
        },
        "saksham_runtime": {
            "suitable": True,
            "confidence": "medium",
            "notes": (
                "Runtime recognizer is not yet wired to ONNX/GRU; integration is a separate Phase 5 task. "
                "Live webcam domain shift from dictionary videos remains a validation risk."
            ),
        },
        "strengths": strengths,
        "risks": risks,
    }


def audit_cislr_dataset(
    dataset_root: str,
    metadata_path: Optional[str] = None,
    label_column: Optional[str] = None,
    top_n: int = 100,
    video_sample_size: int = 25,
) -> Dict[str, Any]:
    resolved_root = Path(dataset_root)
    resolved_metadata = metadata_path or str(resolved_root / "dataset.csv")
    if not os.path.exists(resolved_metadata):
        raise FileNotFoundError(
            f"CISLR metadata CSV not found: {resolved_metadata}. "
            "Run: python tools/cislr/download_dataset.py"
        )

    fieldnames, rows = load_csv_rows(resolved_metadata)
    resolved_label_column = detect_label_column(fieldnames, preferred=label_column)
    signer_column = detect_signer_column(fieldnames)
    video_dir = discover_video_root(str(resolved_root))
    video_resolver = VideoPathResolver(video_dir) if video_dir else None
    video_column = detect_video_column(
        fieldnames,
        rows,
        video_dir,
        resolver=video_resolver,
    )

    gloss_counter: Counter = Counter()
    normalized_counter: Counter = Counter()
    saksham_counter: Counter = Counter()
    signer_counter: Counter = Counter()
    source_counter: Counter = Counter()

    for row in rows:
        raw_label = row.get(resolved_label_column, "")
        normalized = canonicalize_label(raw_label)
        gloss_counter[normalized] += 1
        normalized_counter[normalized] += 1
        saksham_label = map_to_saksham_label(raw_label)
        if saksham_label:
            saksham_counter[saksham_label] += 1
        if signer_column:
            signer_value = row.get(signer_column, "").strip() or "UNKNOWN_SIGNER"
            signer_counter[signer_value] += 1
        for source_key in ("source", "dataset", "origin", "provider"):
            if source_key in {name.lower(): name for name in fieldnames}:
                column = {name.lower(): name for name in fieldnames}[source_key]
                source_value = row.get(column, "").strip()
                if source_value:
                    source_counter[source_value] += 1

    gloss_distribution = _distribution_summary(gloss_counter)
    top_signs = [
        {"gloss": gloss, "count": count}
        for gloss, count in gloss_counter.most_common(top_n)
    ]

    video_paths = _sample_video_paths(rows, video_column, video_resolver, video_sample_size)
    video_inspections = [_inspect_video_file(path) for path in video_paths]
    frame_counts = [entry["frame_count"] for entry in video_inspections if entry["exists"]]
    durations = [entry["duration_sec"] for entry in video_inspections if entry["exists"]]
    widths = {entry["width"] for entry in video_inspections if entry["width"]}
    heights = {entry["height"] for entry in video_inspections if entry["height"]}
    fps_values = {entry["fps"] for entry in video_inspections if entry["fps"]}

    missing_videos = 0
    matched_videos = 0
    projected_landmark_bytes = 0
    duration_column = next(
        (name for name in fieldnames if name.lower() == "duration"),
        None,
    )
    if video_dir and video_column and video_resolver:
        seen_paths = set()
        for row in rows:
            resolved_video = video_resolver.resolve(row.get(video_column, ""))
            if not resolved_video or resolved_video in seen_paths:
                continue
            seen_paths.add(resolved_video)
            if os.path.exists(resolved_video):
                matched_videos += 1
                if duration_column:
                    try:
                        frame_estimate = max(int(float(row.get(duration_column, 0)) * 25), 1)
                    except (TypeError, ValueError):
                        frame_estimate = get_video_frame_count(resolved_video)
                else:
                    frame_estimate = get_video_frame_count(resolved_video)
                projected_landmark_bytes += frame_estimate * BYTES_PER_FRAME
            else:
                missing_videos += 1

    metadata_bytes = os.path.getsize(resolved_metadata)
    total_video_bytes = sum_directory_video_bytes(video_dir)

    signer_summary = {
        "signer_column": signer_column or "",
        "unique_signers": len(signer_counter),
        "top_signers": [
            {"signer_id": signer, "count": count}
            for signer, count in signer_counter.most_common(20)
        ],
    }

    storage_layout = {
        "dataset_root": str(resolved_root),
        "metadata_path": resolved_metadata,
        "metadata_bytes": metadata_bytes,
        "video_dir": video_dir or "",
        "video_search_root": video_dir or "",
        "indexed_videos": video_resolver.indexed_video_count if video_resolver else 0,
        "video_column": video_column or "",
        "label_column": resolved_label_column,
        "csv_columns": fieldnames,
        "auxiliary_files": {
            "prototype_csv": os.path.exists(str(resolved_root / "prototype.csv")),
            "test_csv": os.path.exists(str(resolved_root / "test.csv")),
            "i3d_features_pkl": os.path.exists(str(resolved_root / "I3D_features.pkl")),
            "video_zip": any(resolved_root.rglob("CISLR_v1.5-a_videos.zip")),
        },
        "total_video_files_on_disk": count_mp4_files(video_dir) if video_dir else 0,
        "total_raw_video_bytes": total_video_bytes,
        "matched_metadata_videos": matched_videos,
        "missing_metadata_videos": missing_videos,
        "projected_landmark_tensor_bytes": projected_landmark_bytes,
        "recommended_working_bytes": metadata_bytes + total_video_bytes + projected_landmark_bytes,
    }

    video_summary = {
        "sampled_videos": len(video_inspections),
        "unique_resolutions": sorted(f"{w}x{h}" for w, h in zip(sorted(widths), sorted(heights))),
        "unique_fps": sorted(fps_values),
        "frame_count_min": min(frame_counts) if frame_counts else 0,
        "frame_count_max": max(frame_counts) if frame_counts else 0,
        "frame_count_avg": (sum(frame_counts) / len(frame_counts)) if frame_counts else 0.0,
        "avg_duration_sec": (sum(durations) / len(durations)) if durations else 0.0,
        "sample_inspections": video_inspections[:5],
    }

    saksham_overlap = {label: saksham_counter.get(label, 0) for label in TARGET_LABELS}
    saksham_target_details = _build_saksham_target_details(
        rows=rows,
        label_column=resolved_label_column,
        video_column=video_column,
        video_resolver=video_resolver,
    )
    vocabulary_tiers = _build_vocabulary_tiers(gloss_counter)
    pipeline = _assess_pipeline_suitability(
        gloss_distribution=gloss_distribution,
        signer_summary=signer_summary,
        video_summary=video_summary,
        saksham_overlap=saksham_overlap,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(resolved_root),
        "metadata_path": resolved_metadata,
        "totals": {
            "rows": len(rows),
            "unique_glosses": len(gloss_counter),
            "unique_signers": signer_summary["unique_signers"],
        },
        "gloss_distribution": gloss_distribution,
        "top_signs": top_signs,
        "signer_summary": signer_summary,
        "source_summary": {
            "unique_sources": len(source_counter),
            "top_sources": [
                {"source": source, "count": count}
                for source, count in source_counter.most_common(10)
            ],
        },
        "saksham_target_overlap": saksham_overlap,
        "saksham_target_details": saksham_target_details,
        "vocabulary_tiers": vocabulary_tiers,
        "storage_layout": storage_layout,
        "video_summary": video_summary,
        "pipeline_suitability": pipeline,
        "prototype_test_splits": {
            "prototype_csv_rows": _count_csv_rows(str(resolved_root / "prototype.csv")),
            "test_csv_rows": _count_csv_rows(str(resolved_root / "test.csv")),
        },
    }


def _count_csv_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    _, rows = load_csv_rows(path)
    return len(rows)


def _build_saksham_target_details(
    rows: Sequence[Dict[str, str]],
    label_column: str,
    video_column: Optional[str],
    video_resolver: Optional[VideoPathResolver],
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        label: {
            "row_count": 0,
            "videos_found": 0,
            "videos_missing": 0,
            "sample_uids": [],
        }
        for label in TARGET_LABELS
    }

    for row in rows:
        saksham_label = map_to_saksham_label(row.get(label_column, ""))
        if not saksham_label:
            continue

        entry = details[saksham_label]
        entry["row_count"] += 1
        uid = row.get(video_column or "uid", "")
        if len(entry["sample_uids"]) < 5 and uid:
            entry["sample_uids"].append(uid)

        if video_resolver and video_column:
            if video_resolver.exists(row.get(video_column, "")):
                entry["videos_found"] += 1
            else:
                entry["videos_missing"] += 1

    missing_targets = [label for label in TARGET_LABELS if details[label]["row_count"] == 0]
    return {
        "targets": details,
        "missing_targets": missing_targets,
        "matched_target_count": sum(1 for label in TARGET_LABELS if details[label]["row_count"] > 0),
    }


def _build_vocabulary_tiers(gloss_counter: Counter) -> Dict[str, Any]:
    tiers = []
    for spec in V1_VOCABULARY_TIERS:
        eligible = [
            {"gloss": gloss, "count": count}
            for gloss, count in gloss_counter.items()
            if count >= spec["min_samples_per_gloss"]
        ]
        eligible.sort(key=lambda item: (-item["count"], item["gloss"]))
        tiers.append(
            {
                "name": spec["name"],
                "min_samples_per_gloss": spec["min_samples_per_gloss"],
                "eligible_gloss_count": len(eligible),
                "recommended_v1_size": spec["recommended_v1_size"],
                "recommended_v1_feasible": len(eligible) >= spec["recommended_v1_size"],
                "top_glosses": eligible[:20],
            }
        )
    return {"tiers": tiers}


def build_vocabulary_validation_report(audit_report: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a SAKSHAM-focused vocabulary validation report from a full audit."""
    saksham = audit_report.get("saksham_target_details", {})
    tiers = audit_report.get("vocabulary_tiers", {})
    storage = audit_report.get("storage_layout", {})

    return {
        "generated_at": audit_report.get("generated_at"),
        "dataset_root": audit_report.get("dataset_root"),
        "video_search_root": storage.get("video_search_root", ""),
        "indexed_videos": storage.get("indexed_videos", 0),
        "matched_metadata_videos": storage.get("matched_metadata_videos", 0),
        "missing_metadata_videos": storage.get("missing_metadata_videos", 0),
        "unique_glosses": audit_report.get("totals", {}).get("unique_glosses", 0),
        "saksham_targets": saksham.get("targets", {}),
        "saksham_missing_targets": saksham.get("missing_targets", []),
        "saksham_matched_target_count": saksham.get("matched_target_count", 0),
        "vocabulary_tiers": tiers.get("tiers", []),
        "recommended_v1_vocabulary_size": 50,
        "recommended_min_samples_per_class": 2,
    }


def render_vocabulary_validation_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# CISLR Vocabulary Validation Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Video search root: `{report.get('video_search_root', '')}`",
        f"Indexed MP4 files: **{report.get('indexed_videos', 0):,}**",
        f"Metadata rows with videos found: **{report.get('matched_metadata_videos', 0):,}**",
        f"Metadata rows missing videos: **{report.get('missing_metadata_videos', 0):,}**",
        f"Unique glosses: **{report.get('unique_glosses', 0):,}**",
        "",
        "## SAKSHAM Target Sign Counts",
        "",
        "| Sign | Rows | Videos Found | Videos Missing | Sample UIDs |",
        "|------|------|--------------|----------------|-------------|",
    ]

    for label in TARGET_LABELS:
        entry = report.get("saksham_targets", {}).get(label, {})
        uids = ", ".join(entry.get("sample_uids", [])[:3]) or "—"
        lines.append(
            f"| {label} | {entry.get('row_count', 0)} | "
            f"{entry.get('videos_found', 0)} | {entry.get('videos_missing', 0)} | {uids} |"
        )

    if report.get("saksham_missing_targets"):
        lines.extend(["", "## Missing SAKSHAM Targets", ""])
        for label in report["saksham_missing_targets"]:
            lines.append(f"- {label}")

    lines.extend(["", "## Vocabulary Tiers", ""])
    for tier in report.get("vocabulary_tiers", []):
        feasible = "feasible" if tier.get("recommended_v1_feasible") else "insufficient"
        lines.append(
            f"- **{tier['name']}** (>={tier['min_samples_per_gloss']} samples/gloss): "
            f"{tier['eligible_gloss_count']:,} eligible glosses; "
            f"V1 target {tier['recommended_v1_size']} → {feasible}"
        )

    lines.extend(
        [
            "",
            "## V1 Recommendation",
            f"- Recommended vocabulary size: **{report.get('recommended_v1_vocabulary_size', 50)}**",
            f"- Minimum samples per class: **{report.get('recommended_min_samples_per_class', 2)}**",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_markdown_report(report: Dict[str, Any]) -> str:
    totals = report["totals"]
    dist = report["gloss_distribution"]
    storage = report["storage_layout"]
    video = report["video_summary"]
    pipeline = report["pipeline_suitability"]

    lines = [
        "# CISLR Dataset Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Dataset root: `{report['dataset_root']}`",
        "",
        "## Totals",
        f"- Rows: **{totals['rows']:,}**",
        f"- Unique glosses: **{totals['unique_glosses']:,}**",
        f"- Unique signers: **{totals['unique_signers']:,}**",
        "",
        "## Class Distribution",
        f"- Min / max samples per gloss: {dist['min']} / {dist['max']}",
        f"- Mean / median samples per gloss: {dist['mean']:.2f} / {dist['median']:.0f}",
        f"- Singleton glosses: {dist['singleton_classes']:,}",
        f"- Glosses with >=2 samples: {dist['classes_with_2_or_more']:,}",
        f"- Glosses with >=5 samples: {dist['classes_with_5_or_more']:,}",
        f"- Glosses with >=10 samples: {dist['classes_with_10_or_more']:,}",
        "",
        "## Top 20 Signs",
    ]

    for entry in report["top_signs"][:20]:
        lines.append(f"- `{entry['gloss']}`: {entry['count']}")

    lines.extend(
        [
            "",
            "## SAKSHAM Target Overlap",
        ]
    )
    for label, count in report["saksham_target_overlap"].items():
        lines.append(f"- {label}: {count}")

    lines.extend(
        [
            "",
            "## Storage Layout",
            f"- Video search root: `{storage.get('video_search_root', storage.get('video_dir', ''))}`",
            f"- Indexed MP4 files: {storage.get('indexed_videos', storage.get('total_video_files_on_disk', 0))}",
            f"- Metadata: {human_bytes(storage['metadata_bytes'])}",
            f"- Raw videos on disk: {human_bytes(storage['total_raw_video_bytes'])} ({storage['total_video_files_on_disk']} files)",
            f"- Matched metadata videos: {storage['matched_metadata_videos']}",
            f"- Missing metadata videos: {storage['missing_metadata_videos']}",
            f"- Projected landmark tensors: {human_bytes(storage['projected_landmark_tensor_bytes'])}",
            f"- Recommended working space: {human_bytes(storage['recommended_working_bytes'])}",
            "",
            "## Video Format (sampled)",
            f"- Sampled videos: {video['sampled_videos']}",
            f"- Resolutions: {', '.join(video['unique_resolutions']) or 'n/a'}",
            f"- FPS values: {', '.join(str(v) for v in video['unique_fps']) or 'n/a'}",
            f"- Frame count range: {video['frame_count_min']} – {video['frame_count_max']}",
            f"- Avg duration (sample): {video['avg_duration_sec']:.2f}s",
            "",
            "## Pipeline Suitability",
        ]
    )

    for stage, assessment in pipeline.items():
        if not isinstance(assessment, dict) or "suitable" not in assessment:
            continue
        status = "YES" if assessment["suitable"] else "NO"
        lines.append(
            f"- **{stage}**: {status} ({assessment['confidence']} confidence) — {assessment['notes']}"
        )

    if pipeline.get("risks"):
        lines.extend(["", "## Risks"])
        for risk in pipeline["risks"]:
            lines.append(f"- {risk}")

    if pipeline.get("strengths"):
        lines.extend(["", "## Strengths"])
        for strength in pipeline["strengths"]:
            lines.append(f"- {strength}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a full CISLR dataset audit.")
    parser.add_argument("--dataset-root", default="cislr_data", help="Root directory of CISLR files.")
    parser.add_argument("--metadata-path", help="Optional explicit path to dataset.csv.")
    parser.add_argument("--label-column", help="Optional explicit gloss column name.")
    parser.add_argument("--top-n", type=int, default=100, help="Number of top signs to include.")
    parser.add_argument("--video-sample-size", type=int, default=25, help="Videos to inspect with OpenCV.")
    parser.add_argument("--json-output", help="Optional JSON report path.")
    parser.add_argument("--markdown-output", help="Optional Markdown report path.")
    args = parser.parse_args()

    ensure_dirs()
    report = audit_cislr_dataset(
        dataset_root=args.dataset_root,
        metadata_path=args.metadata_path,
        label_column=args.label_column,
        top_n=args.top_n,
        video_sample_size=args.video_sample_size,
    )

    json_output = args.json_output or str(REPORTS_DIR / "cislr_audit.json")
    markdown_output = args.markdown_output or str(REPORTS_DIR / "cislr_audit.md")

    os.makedirs(os.path.dirname(json_output), exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    markdown = render_markdown_report(report)
    with open(markdown_output, "w", encoding="utf-8") as handle:
        handle.write(markdown)

    vocab_report = build_vocabulary_validation_report(report)
    vocab_json = str(REPORTS_DIR / "vocabulary_validation.json")
    vocab_md = str(REPORTS_DIR / "vocabulary_validation.md")
    with open(vocab_json, "w", encoding="utf-8") as handle:
        json.dump(vocab_report, handle, indent=2)
    with open(vocab_md, "w", encoding="utf-8") as handle:
        handle.write(render_vocabulary_validation_markdown(vocab_report))

    print(render_markdown_report(report))
    print(f"JSON report: {json_output}")
    print(f"Markdown report: {markdown_output}")
    print(f"Vocabulary validation JSON: {vocab_json}")
    print(f"Vocabulary validation Markdown: {vocab_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
