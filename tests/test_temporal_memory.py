import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from mosaic_lab.temporal_memory import EnrollmentConfig, build_temporal_memory, iter_memory_records


class TemporalMemoryTests(unittest.TestCase):
    def test_memory_is_deterministic_and_preserves_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sr = 22050
            tone_time = np.arange(int(0.8 * sr), dtype=np.float32) / sr
            waveform = np.concatenate(
                [np.zeros(int(0.4 * sr), dtype=np.float32), 0.3 * np.sin(2 * np.pi * 330.0 * tone_time)]
            )
            source = root / "source.wav"
            sf.write(source, waveform, sr)
            output = root / "memory"
            memory_path = build_temporal_memory(
                source,
                output,
                config=EnrollmentConfig(
                    patch_seconds=0.4,
                    hop_seconds=0.4,
                    analysis_sr=sr,
                    min_active_ratio=0.2,
                    min_f0_confidence=0.1,
                ),
            )
            metadata = json.loads(memory_path.read_text(encoding="utf-8"))
            records = list(iter_memory_records(output))
            self.assertEqual(metadata["patch_count"], 3)
            self.assertEqual([record["patch_id"] for record in records], ["patch_000001", "patch_000002", "patch_000003"])
            self.assertFalse(records[0]["accepted"])
            self.assertIn("low_active_ratio", records[0]["rejection_reasons"])
            self.assertTrue(any(record["accepted"] for record in records[1:]))
            expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(metadata["source_sha256"], expected_hash)
            for record in records:
                self.assertTrue((output / record["audio_path"]).is_file())
                self.assertTrue((output / record["feature_path"]).is_file())

    def test_existing_output_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sr = 22050
            time = np.arange(sr, dtype=np.float32) / sr
            source = root / "source.wav"
            sf.write(source, 0.3 * np.sin(2 * np.pi * 220.0 * time), sr)
            output = root / "memory"
            build_temporal_memory(source, output)
            sentinel = output / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_temporal_memory(source, output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
