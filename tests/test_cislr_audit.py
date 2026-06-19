import csv
import os
import tempfile
import unittest

from tools.cislr.audit_dataset import audit_cislr_dataset, render_markdown_report


class TestCISLRAuditDataset(unittest.TestCase):
    def _write_fixture(self, temp_dir: str) -> str:
        video_dir = os.path.join(temp_dir, "CISLR_v1.5-a_videos")
        os.makedirs(video_dir, exist_ok=True)

        csv_path = os.path.join(temp_dir, "dataset.csv")
        rows = [
            {"word": "HELLO", "signer_id": "S1", "video_id": "hello_1"},
            {"word": "HELLO", "signer_id": "S2", "video_id": "hello_2"},
            {"word": "thanks", "signer_id": "S1", "video_id": "thanks_1"},
            {"word": "WATER", "signer_id": "S3", "video_id": "water_1"},
            {"word": "UNKNOWN", "signer_id": "S4", "video_id": "other_1"},
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["word", "signer_id", "video_id"])
            writer.writeheader()
            writer.writerows(rows)

        for filename in ("hello_1.mp4", "hello_2.mp4", "thanks_1.mp4", "water_1.mp4"):
            with open(os.path.join(video_dir, filename), "wb") as handle:
                handle.write(b"\0" * 64)

        return csv_path

    def test_full_audit_counts_and_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_fixture(temp_dir)
            report = audit_cislr_dataset(
                dataset_root=temp_dir,
                top_n=10,
                video_sample_size=3,
            )

            self.assertEqual(report["totals"]["rows"], 5)
            self.assertEqual(report["totals"]["unique_glosses"], 4)
            self.assertEqual(report["totals"]["unique_signers"], 4)
            self.assertEqual(report["gloss_distribution"]["max"], 2)
            self.assertEqual(report["saksham_target_overlap"]["HELLO"], 2)
            self.assertEqual(report["saksham_target_overlap"]["THANK YOU"], 1)
            self.assertEqual(report["saksham_target_overlap"]["WATER"], 1)
            self.assertTrue(report["pipeline_suitability"]["mediapipe_landmarks"]["suitable"])
            self.assertEqual(report["storage_layout"]["matched_metadata_videos"], 4)

    def test_markdown_report_contains_headings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_fixture(temp_dir)
            report = audit_cislr_dataset(dataset_root=temp_dir, video_sample_size=1)
            markdown = render_markdown_report(report)
            self.assertIn("# CISLR Dataset Audit", markdown)
            self.assertIn("## Class Distribution", markdown)
            self.assertIn("## Pipeline Suitability", markdown)


if __name__ == "__main__":
    unittest.main()
