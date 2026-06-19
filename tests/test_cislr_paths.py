"""Tests for nested CISLR video path discovery."""
import csv
import os
import tempfile
import unittest

from tools.cislr.audit_dataset import audit_cislr_dataset
from tools.cislr.paths import VideoPathResolver, discover_video_root


class TestNestedVideoPaths(unittest.TestCase):
    def test_discover_nested_video_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            nested = os.path.join(temp_dir, "CISLR_v1.5-a_videos", "CISLR_v1.5-a_videos")
            os.makedirs(nested, exist_ok=True)
            with open(os.path.join(nested, "sample.mp4"), "wb") as handle:
                handle.write(b"\0" * 32)

            video_root = discover_video_root(temp_dir)
            self.assertEqual(video_root, nested)

            resolver = VideoPathResolver(video_root)
            self.assertTrue(resolver.exists("sample"))
            self.assertTrue(resolver.resolve("sample").endswith("sample.mp4"))

    def test_audit_finds_nested_videos(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            nested = os.path.join(temp_dir, "CISLR_v1.5-a_videos", "CISLR_v1.5-a_videos")
            os.makedirs(nested, exist_ok=True)

            csv_path = os.path.join(temp_dir, "dataset.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["uid", "gloss", "duration"])
                writer.writeheader()
                writer.writerows([
                    {"uid": "hello_1", "gloss": "HELLO", "duration": "4"},
                    {"uid": "thanks_1", "gloss": "thanks", "duration": "3"},
                ])

            for uid in ("hello_1", "thanks_1"):
                with open(os.path.join(nested, f"{uid}.mp4"), "wb") as handle:
                    handle.write(b"\0" * 64)

            report = audit_cislr_dataset(dataset_root=temp_dir, video_sample_size=1)
            self.assertEqual(report["storage_layout"]["matched_metadata_videos"], 2)
            self.assertEqual(report["storage_layout"]["total_video_files_on_disk"], 2)
            self.assertEqual(report["storage_layout"]["indexed_videos"], 2)


if __name__ == "__main__":
    unittest.main()
