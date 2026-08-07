import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from mosaic_lab.temporal_features import (
    analyze_temporal_audio,
    extract_patch_features,
    iter_patch_bounds,
    json_safe,
)


class TemporalFeatureTests(unittest.TestCase):
    def _write(self, root: Path, name: str, waveform: np.ndarray, sr: int = 22050) -> Path:
        path = root / name
        sf.write(path, waveform, sr)
        return path

    def test_sine_has_valid_f0_and_finite_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sr = 22050
            time = np.arange(sr, dtype=np.float32) / sr
            path = self._write(root, "tone.wav", 0.3 * np.sin(2 * np.pi * 440.0 * time))
            analysis = analyze_temporal_audio(path, analysis_sr=sr)
            bounds = next(iter_patch_bounds(analysis, patch_seconds=0.4, hop_seconds=0.1))
            features = extract_patch_features(analysis, start_sample=bounds[0], end_sample=bounds[1])
            self.assertTrue(features.key.f0_valid)
            self.assertAlmostEqual(features.key.f0_median_hz, 440.0, delta=8.0)
            self.assertGreater(features.key.voiced_ratio, 0.5)
            payload = json.dumps(json_safe(asdict(features.key)), allow_nan=False)
            self.assertNotIn("NaN", payload)

    def test_silence_is_unvoiced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = self._write(Path(temp), "silence.wav", np.zeros(22050, dtype=np.float32))
            analysis = analyze_temporal_audio(path)
            bounds = next(iter_patch_bounds(analysis, patch_seconds=0.4, hop_seconds=0.1))
            features = extract_patch_features(analysis, start_sample=bounds[0], end_sample=bounds[1])
            self.assertFalse(features.key.f0_valid)
            self.assertLess(features.key.voiced_ratio, 0.1)
            self.assertLess(features.quality.active_ratio, 0.1)

    def test_clipping_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            waveform = np.ones(22050, dtype=np.float32)
            path = self._write(Path(temp), "clipped.wav", waveform, sr=22050)
            analysis = analyze_temporal_audio(path)
            bounds = next(iter_patch_bounds(analysis, patch_seconds=0.4, hop_seconds=0.1))
            features = extract_patch_features(analysis, start_sample=bounds[0], end_sample=bounds[1])
            self.assertGreater(features.quality.clipping_ratio, 0.9)

    def test_json_safe_replaces_nonfinite_values(self) -> None:
        payload = json_safe({"nan": math.nan, "inf": math.inf, "nested": [math.nan]})
        json.dumps(payload, allow_nan=False)
        self.assertEqual(payload["nan"], 0.0)


if __name__ == "__main__":
    unittest.main()
