import json
import tempfile
import unittest
from pathlib import Path

from mosaic_lab.temporal_visualize import visualize_temporal_query


class TemporalVisualizeTests(unittest.TestCase):
    def test_empty_query_creates_html_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory = root / "memory"
            memory.mkdir()
            (memory / "memory.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "memory_type": "mosaic_temporal_timbre_memory",
                        "patch_count": 0,
                        "accepted_patch_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            (memory / "memory.jsonl").write_text("", encoding="utf-8")
            query = root / "query.jsonl"
            query.write_text("", encoding="utf-8")
            output = root / "nested" / "report"
            paths = visualize_temporal_query(query, memory, output)
            self.assertTrue(Path(paths["html"]).is_file())
            self.assertTrue(Path(paths["summary"]).is_file())
            summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["query_frames"], 0)

    def test_unknown_output_suffix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory = root / "memory"
            memory.mkdir()
            (memory / "memory.json").write_text(
                json.dumps({"schema_version": 1, "memory_type": "mosaic_temporal_timbre_memory"}),
                encoding="utf-8",
            )
            (memory / "memory.jsonl").write_text("", encoding="utf-8")
            query = root / "query.jsonl"
            query.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                visualize_temporal_query(query, memory, root / "report.txt")


if __name__ == "__main__":
    unittest.main()
