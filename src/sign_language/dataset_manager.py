import os
import csv
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetManager:
    """
    Manages the collection, storage, and export of sign language dataset sequences.
    Records temporal sequences of hand landmarks for future dynamic sign language model training.
    """
    MIN_FRAME_COUNT = 15
    RECOMMENDED_FRAME_COUNT = 24
    MIN_ACTIVE_HAND_PRESENCE_RATIO = 0.60
    RECOMMENDED_ACTIVE_HAND_PRESENCE_RATIO = 0.85
    MIN_CLIPS_PER_LABEL_TARGET = 40
    MIN_SIGNERS_TARGET = 5

    def __init__(self, data_dir: str = "dataset", export_dir: str = "exports"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

        self.accepted_dir = os.path.join(self.data_dir, "accepted")
        self.rejected_dir = os.path.join(self.data_dir, "rejected")
        os.makedirs(self.accepted_dir, exist_ok=True)
        os.makedirs(self.rejected_dir, exist_ok=True)
            
        self.is_recording = False
        self.current_label = ""
        self.current_signer_id = ""
        self.current_sequence_id = ""
        self.current_sequence = []
        self.review_clip: Optional[Dict] = None
        self.manifest_path = os.path.join(self.data_dir, "manifest.jsonl")
        
        # Meta storage for multiple sequences
        self.session_sequences = []

    def start_recording(self, label: str, signer_id: str = "SIGNER_01"):
        """Starts recording a new temporal sequence of landmarks for the given sign label."""
        if self.is_recording:
            self.stop_recording()
            
        self.current_label = label.upper()
        self.current_signer_id = signer_id
        self.current_sequence_id = f"{self.current_label}_{int(time.time() * 1000)}"
        self.current_sequence = []
        self.is_recording = True
        self.review_clip = None
        logging.info(f"DatasetManager: Started recording sequence for '{self.current_label}' ({self.current_signer_id})")

    @staticmethod
    def _hand_entry(hand: Dict) -> Dict:
        if not hand or not hand.get("present"):
            return {
                "present": False,
                "confidence": 0.0,
                "landmarks": [[0.0, 0.0, 0.0] for _ in range(21)],
            }
        return {
            "present": True,
            "confidence": float(hand.get("confidence", 0.0)),
            "landmarks": [list(pt) for pt in hand.get("landmarks", [])],
        }

    def add_frame_data(self, hand_data: Dict):
        """Adds a single frame's hand landmark data to the current sequence."""
        if not self.is_recording or not hand_data:
            return

        left_hand = self._hand_entry(hand_data.get("left_hand", {}))
        right_hand = self._hand_entry(hand_data.get("right_hand", {}))
        frame_entry = {
            "timestamp": time.time(),
            "sequence_id": self.current_sequence_id,
            "label": self.current_label,
            "signer_id": self.current_signer_id,
            "left_hand": left_hand,
            "right_hand": right_hand,
        }
        self.current_sequence.append(frame_entry)

    def stop_recording(self) -> Optional[Dict]:
        """Stops the recording and prepares the clip for accept/reject review."""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        if len(self.current_sequence) > 0:
            review_clip = {
                "sequence_id": self.current_sequence_id,
                "label": self.current_label,
                "signer_id": self.current_signer_id,
                "frames": self.current_sequence,
            }
            self.review_clip = review_clip
            logging.info(f"DatasetManager: Stopped recording. Sequence '{self.current_sequence_id}' has {len(self.current_sequence)} frames.")
            self.current_sequence = []
            return self.get_review_summary()
        self.current_sequence = []
        return None

    def has_review_clip(self) -> bool:
        return self.review_clip is not None

    def _summarize_frames(self, frames: List[Dict]) -> Dict:
        frame_count = len(frames)
        left_present = sum(1 for frame in frames if frame["left_hand"]["present"])
        right_present = sum(1 for frame in frames if frame["right_hand"]["present"])
        both_hands_present = sum(
            1 for frame in frames if frame["left_hand"]["present"] and frame["right_hand"]["present"]
        )
        left_presence_ratio = (left_present / frame_count) if frame_count else 0.0
        right_presence_ratio = (right_present / frame_count) if frame_count else 0.0
        active_hand_ratio = max(left_presence_ratio, right_presence_ratio)
        both_hands_ratio = (both_hands_present / frame_count) if frame_count else 0.0

        blocking_issues = []
        warnings = []
        if frame_count < self.MIN_FRAME_COUNT:
            blocking_issues.append(
                f"Need at least {self.MIN_FRAME_COUNT} frames; recorded {frame_count}."
            )
        elif frame_count < self.RECOMMENDED_FRAME_COUNT:
            warnings.append(
                f"Short clip: {frame_count} frames. Recommended at least {self.RECOMMENDED_FRAME_COUNT}."
            )

        if active_hand_ratio < self.MIN_ACTIVE_HAND_PRESENCE_RATIO:
            blocking_issues.append(
                f"Low hand visibility: best hand tracked in {active_hand_ratio:.0%} of frames; minimum is {self.MIN_ACTIVE_HAND_PRESENCE_RATIO:.0%}."
            )
        elif active_hand_ratio < self.RECOMMENDED_ACTIVE_HAND_PRESENCE_RATIO:
            warnings.append(
                f"Hand visibility is only {active_hand_ratio:.0%}. Recommended at least {self.RECOMMENDED_ACTIVE_HAND_PRESENCE_RATIO:.0%}."
            )

        return {
            "frame_count": frame_count,
            "left_presence_ratio": left_presence_ratio,
            "right_presence_ratio": right_presence_ratio,
            "active_hand_ratio": active_hand_ratio,
            "both_hands_ratio": both_hands_ratio,
            "quality_blockers": blocking_issues,
            "quality_warnings": warnings,
            "passes_quality_checks": len(blocking_issues) == 0,
        }

    def get_review_summary(self) -> Optional[Dict]:
        if not self.review_clip:
            return None

        summary = {
            "sequence_id": self.review_clip["sequence_id"],
            "label": self.review_clip["label"],
            "signer_id": self.review_clip["signer_id"],
        }
        summary.update(self._summarize_frames(self.review_clip["frames"]))
        return summary

    def _save_clip_metadata(self, target_dir: str, metadata: Dict):
        metadata_path = os.path.join(target_dir, f"{metadata['sample_id']}.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def _save_clip_arrays(self, target_dir: str, sample_id: str, frames: List[Dict]):
        timestamps = np.asarray([frame["timestamp"] for frame in frames], dtype=np.float32)
        left_hand = np.asarray([frame["left_hand"].get("isometric_landmarks", frame["left_hand"].get("landmarks", [])) for frame in frames], dtype=np.float32)
        right_hand = np.asarray([frame["right_hand"].get("isometric_landmarks", frame["right_hand"].get("landmarks", [])) for frame in frames], dtype=np.float32)
        left_present = np.asarray([float(frame["left_hand"]["present"]) for frame in frames], dtype=np.float32)
        right_present = np.asarray([float(frame["right_hand"]["present"]) for frame in frames], dtype=np.float32)
        left_confidence = np.asarray([frame["left_hand"]["confidence"] for frame in frames], dtype=np.float32)
        right_confidence = np.asarray([frame["right_hand"]["confidence"] for frame in frames], dtype=np.float32)
        np.savez_compressed(
            os.path.join(target_dir, f"{sample_id}.npz"),
            timestamps=timestamps,
            left_hand=left_hand,
            right_hand=right_hand,
            left_present=left_present,
            right_present=right_present,
            left_confidence=left_confidence,
            right_confidence=right_confidence,
        )

    def _build_metadata(self, summary: Dict, status: str, reason: Optional[str] = None) -> Dict:
        metadata = {
            "sample_id": summary["sequence_id"],
            "label": summary["label"],
            "signer_id": summary["signer_id"],
            "frame_count": summary["frame_count"],
            "left_presence_ratio": summary["left_presence_ratio"],
            "right_presence_ratio": summary["right_presence_ratio"],
            "active_hand_ratio": summary["active_hand_ratio"],
            "both_hands_ratio": summary["both_hands_ratio"],
            "quality_warnings": summary["quality_warnings"],
            "quality_blockers": summary["quality_blockers"],
            "passes_quality_checks": summary["passes_quality_checks"],
            "saved_at": time.time(),
            "status": status,
        }
        if reason:
            metadata["reason"] = reason
        return metadata

    def accept_current_clip(self) -> Optional[Dict]:
        if not self.review_clip:
            return None

        summary = self.get_review_summary()
        if not summary["passes_quality_checks"]:
            blocked = self._build_metadata(summary, status="blocked")
            blocked["message"] = "Clip does not meet dataset quality thresholds."
            return blocked

        sample_id = self.review_clip["sequence_id"]
        metadata = self._build_metadata(summary, status="accepted")
        self._save_clip_arrays(self.accepted_dir, sample_id, self.review_clip["frames"])
        self._save_clip_metadata(self.accepted_dir, metadata)
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")

        self.session_sequences.append(metadata)
        self.review_clip = None
        return metadata

    def reject_current_clip(self, reason: str = "Rejected") -> Optional[Dict]:
        if not self.review_clip:
            return None

        summary = self.get_review_summary()
        metadata = self._build_metadata(summary, status="rejected", reason=reason)
        self._save_clip_metadata(self.rejected_dir, metadata)
        self.review_clip = None
        return metadata

    def _load_manifest_entries(self) -> List[Dict]:
        if not os.path.exists(self.manifest_path):
            return []

        entries = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def get_collection_targets(self, expected_labels: Optional[List[str]] = None) -> Dict:
        labels = [label.upper() for label in (expected_labels or [])]
        label_count = len(labels)
        return {
            "minimum_clips_per_sign": self.MIN_CLIPS_PER_LABEL_TARGET,
            "minimum_signers": self.MIN_SIGNERS_TARGET,
            "target_dataset_size": label_count * self.MIN_CLIPS_PER_LABEL_TARGET if label_count else 0,
            "expected_label_count": label_count,
            "expected_labels": labels,
        }

    def get_dataset_statistics(self, expected_labels: Optional[List[str]] = None) -> Dict:
        records = self._load_manifest_entries()
        labels = [label.upper() for label in (expected_labels or [])]
        clips_per_label: Dict[str, int] = {label: 0 for label in labels}
        clips_per_signer: Dict[str, int] = {}
        signers_per_label: Dict[str, set] = {label: set() for label in labels}

        frame_counts: List[float] = []
        left_ratios: List[float] = []
        right_ratios: List[float] = []
        active_ratios: List[float] = []
        both_ratios: List[float] = []

        for record in records:
            label = record.get("label", "").upper()
            signer_id = record.get("signer_id", "")
            clips_per_label[label] = clips_per_label.get(label, 0) + 1
            clips_per_signer[signer_id] = clips_per_signer.get(signer_id, 0) + 1
            signers_per_label.setdefault(label, set()).add(signer_id)

            frame_counts.append(float(record.get("frame_count", 0)))
            left_ratios.append(float(record.get("left_presence_ratio", 0.0)))
            right_ratios.append(float(record.get("right_presence_ratio", 0.0)))
            active_ratios.append(float(record.get("active_hand_ratio", 0.0)))
            both_ratios.append(float(record.get("both_hands_ratio", 0.0)))

        nonzero_label_counts = {label: count for label, count in clips_per_label.items() if count > 0}
        missing_labels = [label for label in labels if clips_per_label.get(label, 0) == 0]
        under_target_labels = [
            label for label, count in clips_per_label.items()
            if count < self.MIN_CLIPS_PER_LABEL_TARGET
        ]
        signer_counts_per_label = {
            label: len(signers) for label, signers in signers_per_label.items()
        }
        labels_below_signer_target = [
            label for label, signer_count in signer_counts_per_label.items()
            if label in clips_per_label and clips_per_label.get(label, 0) > 0 and signer_count < self.MIN_SIGNERS_TARGET
        ]

        min_count = min(nonzero_label_counts.values()) if nonzero_label_counts else 0
        max_count = max(nonzero_label_counts.values()) if nonzero_label_counts else 0
        min_label = min(nonzero_label_counts, key=nonzero_label_counts.get) if nonzero_label_counts else ""
        max_label = max(nonzero_label_counts, key=nonzero_label_counts.get) if nonzero_label_counts else ""
        targets = self.get_collection_targets(labels)
        target_dataset_size = targets["target_dataset_size"]
        overall_progress = (
            min(len(records) / target_dataset_size, 1.0) if target_dataset_size else 0.0
        )

        return {
            "total_clips": len(records),
            "clips_per_label": dict(sorted(clips_per_label.items())),
            "clips_per_signer": dict(sorted(clips_per_signer.items())),
            "signers_per_label": dict(sorted(
                (label, len(signers)) for label, signers in signers_per_label.items()
            )),
            "label_signer_coverage": dict(sorted(
                (label, sorted(signers)) for label, signers in signers_per_label.items()
            )),
            "unique_signer_count": len(clips_per_signer),
            "average_frame_count": self._safe_mean(frame_counts),
            "min_frame_count": int(min(frame_counts)) if frame_counts else 0,
            "max_frame_count": int(max(frame_counts)) if frame_counts else 0,
            "hand_visibility_metrics": {
                "left_presence_avg": self._safe_mean(left_ratios),
                "right_presence_avg": self._safe_mean(right_ratios),
                "active_hand_avg": self._safe_mean(active_ratios),
                "both_hands_avg": self._safe_mean(both_ratios),
            },
            "class_balance": {
                "min_count": min_count,
                "max_count": max_count,
                "min_label": min_label,
                "max_label": max_label,
                "imbalance_ratio": (max_count / min_count) if min_count else 0.0,
                "missing_labels": missing_labels,
                "under_target_labels": sorted(under_target_labels),
                "labels_below_signer_target": sorted(labels_below_signer_target),
            },
            "targets": targets,
            "progress": {
                "clips_progress_ratio": overall_progress,
                "clips_remaining": max(target_dataset_size - len(records), 0),
                "signers_progress_ratio": min(
                    len(clips_per_signer) / self.MIN_SIGNERS_TARGET, 1.0
                ) if self.MIN_SIGNERS_TARGET else 0.0,
            },
        }

    def audit_dataset_quality(self, expected_labels: Optional[List[str]] = None) -> Dict:
        records = self._load_manifest_entries()
        issues = []
        verified_samples = 0
        samples_with_warnings = 0
        samples_with_blockers = 0
        samples_with_both_hands = 0

        for record in records:
            sample_id = record.get("sample_id", "")
            npz_path = os.path.join(self.accepted_dir, f"{sample_id}.npz")
            metadata_path = os.path.join(self.accepted_dir, f"{sample_id}.json")
            sample_issue = {"sample_id": sample_id, "issues": []}

            if not os.path.exists(npz_path):
                sample_issue["issues"].append("Missing .npz data file.")
            if not os.path.exists(metadata_path):
                sample_issue["issues"].append("Missing metadata .json file.")

            if record.get("quality_warnings"):
                samples_with_warnings += 1
            if record.get("quality_blockers"):
                samples_with_blockers += 1
            if float(record.get("both_hands_ratio", 0.0)) > 0.0:
                samples_with_both_hands += 1

            if os.path.exists(npz_path):
                try:
                    with np.load(npz_path) as sample:
                        required_keys = {
                            "timestamps",
                            "left_hand",
                            "right_hand",
                            "left_present",
                            "right_present",
                            "left_confidence",
                            "right_confidence",
                        }
                        missing_keys = sorted(required_keys.difference(sample.files))
                        if missing_keys:
                            sample_issue["issues"].append(
                                f"Missing arrays: {', '.join(missing_keys)}."
                            )
                        else:
                            frame_count = int(record.get("frame_count", 0))
                            if sample["timestamps"].shape[0] != frame_count:
                                sample_issue["issues"].append(
                                    f"Frame count mismatch: manifest={frame_count}, npz={sample['timestamps'].shape[0]}."
                                )
                            if sample["left_hand"].shape != sample["right_hand"].shape:
                                sample_issue["issues"].append("Left/right hand array shapes do not match.")
                            if len(sample["left_hand"].shape) != 3 or sample["left_hand"].shape[1:] != (21, 3):
                                sample_issue["issues"].append("Hand landmark arrays do not match [frames, 21, 3].")
                            if sample["left_present"].shape[0] != sample["timestamps"].shape[0]:
                                sample_issue["issues"].append("Left presence array length mismatch.")
                            if sample["right_present"].shape[0] != sample["timestamps"].shape[0]:
                                sample_issue["issues"].append("Right presence array length mismatch.")
                except Exception as exc:
                    sample_issue["issues"].append(f"Failed to read .npz: {exc}")

            if sample_issue["issues"]:
                issues.append(sample_issue)
            else:
                verified_samples += 1

        statistics = self.get_dataset_statistics(expected_labels)
        return {
            "total_samples": len(records),
            "verified_samples": verified_samples,
            "samples_with_warnings": samples_with_warnings,
            "samples_with_blockers": samples_with_blockers,
            "samples_with_both_hands": samples_with_both_hands,
            "issues": issues,
            "statistics": statistics,
            "ready_for_training": (
                len(records) > 0
                and verified_samples == len(records)
                and not statistics["class_balance"]["missing_labels"]
                and statistics["unique_signer_count"] >= self.MIN_SIGNERS_TARGET
                and statistics["total_clips"] >= statistics["targets"]["target_dataset_size"]
            ),
        }

    def verify_two_hand_capture(self) -> Dict:
        records = self._load_manifest_entries()
        verification = {
            "accepted_samples": len(records),
            "storage_fields_verified": [
                "left_hand",
                "right_hand",
                "left_present",
                "right_present",
                "left_confidence",
                "right_confidence",
            ],
            "samples_with_left_hand": 0,
            "samples_with_right_hand": 0,
            "samples_with_both_hands": 0,
            "verified_sample_id": "",
            "two_hand_capture_verified": False,
            "message": "",
        }

        for record in records:
            sample_id = record.get("sample_id", "")
            npz_path = os.path.join(self.accepted_dir, f"{sample_id}.npz")
            if not os.path.exists(npz_path):
                continue
            with np.load(npz_path) as sample:
                left_present = sample["left_present"]
                right_present = sample["right_present"]
                if float(np.max(left_present)) > 0.0:
                    verification["samples_with_left_hand"] += 1
                if float(np.max(right_present)) > 0.0:
                    verification["samples_with_right_hand"] += 1
                if bool(np.any((left_present > 0.0) & (right_present > 0.0))):
                    verification["samples_with_both_hands"] += 1
                    if not verification["verified_sample_id"]:
                        verification["verified_sample_id"] = sample_id
                        verification["two_hand_capture_verified"] = True

        if verification["two_hand_capture_verified"]:
            verification["message"] = (
                f"Verified two-hand capture in accepted sample {verification['verified_sample_id']}."
            )
        elif records:
            verification["message"] = (
                "Accepted samples exist, but no stored clip currently contains both hands in the same frame."
            )
        else:
            verification["message"] = "No accepted samples available yet to verify two-hand capture."

        return verification

    def get_live_recording_feedback(self) -> Optional[Dict]:
        if not self.is_recording:
            return None

        summary = self._summarize_frames(self.current_sequence)
        frames_remaining = max(self.MIN_FRAME_COUNT - summary["frame_count"], 0)
        recommended_frames_remaining = max(self.RECOMMENDED_FRAME_COUNT - summary["frame_count"], 0)
        last_frame = self.current_sequence[-1] if self.current_sequence else {}
        left_visible = bool(last_frame.get("left_hand", {}).get("present"))
        right_visible = bool(last_frame.get("right_hand", {}).get("present"))

        if frames_remaining > 0:
            feedback = f"Collect {frames_remaining} more frames to meet the minimum clip length."
            quality_state = "needs_frames"
        elif summary["active_hand_ratio"] < self.MIN_ACTIVE_HAND_PRESENCE_RATIO:
            feedback = "Tracking is weak. Keep the active hand clearly visible in the camera."
            quality_state = "blocked"
        elif summary["quality_warnings"]:
            feedback = summary["quality_warnings"][0]
            quality_state = "warning"
        elif recommended_frames_remaining > 0:
            feedback = f"Quality looks acceptable. {recommended_frames_remaining} more frames would strengthen the clip."
            quality_state = "good"
        else:
            feedback = "Quality looks strong. This clip is ready for review when you stop recording."
            quality_state = "excellent"

        summary.update({
            "left_visible_now": left_visible,
            "right_visible_now": right_visible,
            "hands_visible_now": int(left_visible) + int(right_visible),
            "feedback": feedback,
            "quality_state": quality_state,
            "frames_remaining_to_minimum": frames_remaining,
            "frames_remaining_to_recommended": recommended_frames_remaining,
        })
        return summary

    def build_collection_report(self, expected_labels: Optional[List[str]] = None) -> Dict:
        statistics = self.get_dataset_statistics(expected_labels)
        audit = self.audit_dataset_quality(expected_labels)
        two_hand = self.verify_two_hand_capture()
        return {
            "statistics": statistics,
            "audit": audit,
            "two_hand_verification": two_hand,
            "targets": statistics["targets"],
        }

    def export_dataset(self, format: str = "both") -> List[str]:
        """Exports accepted dataset metadata to disk and returns the created file paths."""
        manifest_entries = self._load_manifest_entries()
        records = manifest_entries if manifest_entries else list(self.session_sequences)
        if not records:
            logging.info("DatasetManager: No accepted sequences to export.")
            return []

        export_formats = ("json", "csv") if format == "both" else (format,)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_base = os.path.join(self.export_dir, f"dataset_export_{timestamp}")
        created_files: List[str] = []

        if "json" in export_formats:
            json_path = f"{export_base}.json"
            payload = {
                "generated_at": datetime.now().isoformat(),
                "sample_count": len(records),
                "source": "manifest" if manifest_entries else "session",
                "samples": records,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
            created_files.append(json_path)

        if "csv" in export_formats:
            csv_path = f"{export_base}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = [
                    "sample_id",
                    "label",
                    "signer_id",
                    "frame_count",
                    "left_presence_ratio",
                    "right_presence_ratio",
                    "active_hand_ratio",
                    "both_hands_ratio",
                    "status",
                    "passes_quality_checks",
                    "quality_warnings",
                ]
                writer.writerow(header)

                for seq in records:
                    writer.writerow([
                        seq.get("sample_id", ""),
                        seq.get("label", ""),
                        seq.get("signer_id", ""),
                        seq.get("frame_count", 0),
                        seq.get("left_presence_ratio", 0.0),
                        seq.get("right_presence_ratio", 0.0),
                        seq.get("active_hand_ratio", 0.0),
                        seq.get("both_hands_ratio", 0.0),
                        seq.get("status", ""),
                        seq.get("passes_quality_checks", True),
                        " | ".join(seq.get("quality_warnings", [])),
                    ])
            created_files.append(csv_path)

        logging.info(f"DatasetManager: Exported {len(records)} accepted sequences to {', '.join(created_files)}")
        return created_files
