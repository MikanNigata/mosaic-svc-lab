import unittest

from mosaic_lab.audio import rerank


class AudioRankingTests(unittest.TestCase):
    def test_identity_can_break_quality_tie(self) -> None:
        rows = [
            {"output_path": "a.wav", "f0_corr": 0.99, "cent_rmse": 40, "uv_mismatch": 0.05, "quality_score": 0.9},
            {"output_path": "b.wav", "f0_corr": 0.99, "cent_rmse": 40, "uv_mismatch": 0.05, "quality_score": 0.9},
        ]
        ranked = rerank(rows, identity_scores={"a.wav": 0.6, "b.wav": 0.8})
        self.assertEqual(ranked[0]["output_path"], "b.wav")

    def test_retention_penalizes_pitch_damage(self) -> None:
        rows = [
            {"output_path": "good.wav", "f0_corr": 0.99, "cent_rmse": 40, "uv_mismatch": 0.04, "quality_score": 0.9},
            {"output_path": "bad.wav", "f0_corr": 0.5, "cent_rmse": 400, "uv_mismatch": 0.3, "quality_score": 0.9},
        ]
        ranked = rerank(rows)
        self.assertEqual(ranked[0]["output_path"], "good.wav")


if __name__ == "__main__":
    unittest.main()
