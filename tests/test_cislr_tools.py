import csv
import json
import os
import tempfile
import unittest

from tools.audit_cislr_metadata import audit_cislr_metadata
from tools.count_targets import build_target_count_report
from tools.normalize_labels import canonicalize_label, map_to_saksham_label


class TestCISLRTools(unittest.TestCase):
    def test_label_normalization_and_mapping(self):
        self.assertEqual(canonicalize_label(" thank-you "), "THANK YOU")
        self.assertEqual(map_to_saksham_label("thanks"), "THANK YOU")
        self.assertEqual(map_to_saksham_label("No Sign"), "NO_SIGN")
        self.assertIsNone(map_to_saksham_label("Good Morning"))

    def test_count_targets_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "dataset.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["word", "signer_id", "video_id"])
                writer.writeheader()
                writer.writerows([
                    {"word": "HELLO", "signer_id": "S1", "video_id": "hello_1"},
                    {"word": "thanks", "signer_id": "S1", "video_id": "thanks_1"},
                    {"word": "Good Morning", "signer_id": "S2", "video_id": "other_1"},
                ])

            report = build_target_count_report(csv_path)
            self.assertEqual(report["total_rows"], 3)
            self.assertEqual(report["matched_rows"], 2)
            self.assertEqual(report["target_counts"]["HELLO"], 1)
            self.assertEqual(report["target_counts"]["THANK YOU"], 1)
            self.assertEqual(report["target_counts"]["NO_SIGN"], 0)

    def test_cislr_audit_counts_storage_and_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_dir = os.path.join(temp_dir, "CISLR_v1.5-a_videos", "CISLR_v1.5-a_videos")
            os.makedirs(video_dir, exist_ok=True)

            csv_path = os.path.join(temp_dir, "dataset.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["word", "signer_id", "video_id"])
                writer.writeheader()
                writer.writerows([
                    {"word": "HELLO", "signer_id": "S1", "video_id": "hello_1"},
                    {"word": "PLEASE", "signer_id": "S2", "video_id": "please_1"},
                    {"word": "Unknown", "signer_id": "S3", "video_id": "other_1"},
                ])

            for filename, size in (("hello_1.mp4", 128), ("please_1.mp4", 256), ("other_1.mp4", 512)):
                with open(os.path.join(video_dir, filename), "wb") as handle:
                    handle.write(b"\0" * size)

            report = audit_cislr_metadata(dataset_root=temp_dir)
            self.assertEqual(report["matched_rows"], 2)
            self.assertEqual(report["target_counts"]["HELLO"], 1)
            self.assertEqual(report["target_counts"]["PLEASE"], 1)
            self.assertEqual(report["filtered_video_count"], 2)
            self.assertEqual(report["missing_filtered_videos"], 0)
            self.assertEqual(report["filtered_raw_video_bytes"], 384)
            self.assertEqual(report["total_raw_video_bytes"], 896)
            self.assertEqual(report["unique_signers_in_filtered_set"], 2)
            self.assertEqual(sum(report["expected_split_sizes"].values()), 2)


if __name__ == "__main__":
    unittest.main()
