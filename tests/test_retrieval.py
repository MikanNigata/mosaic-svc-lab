import unittest

from mosaic_lab.retrieval import normalize_weights, rank_prompts


class RetrievalTests(unittest.TestCase):
    def test_matching_prompt_ranks_first(self) -> None:
        source = {
            "register_percentile": 0.8,
            "f0_span_semitones": 12.0,
            "energy_percentile": 0.75,
        }
        prompts = [
            {
                "prompt_id": "far",
                "register_percentile": 0.2,
                "f0_span_semitones": 4.0,
                "energy_percentile": 0.2,
                "quality_score": 1.0,
            },
            {
                "prompt_id": "near",
                "register_percentile": 0.78,
                "f0_span_semitones": 11.5,
                "energy_percentile": 0.72,
                "quality_score": 0.9,
            },
        ]
        ranking = rank_prompts(source, prompts)
        self.assertEqual(ranking[0].prompt_id, "near")
        self.assertGreater(ranking[0].score, ranking[1].score)

    def test_weights_are_normalized(self) -> None:
        weights = normalize_weights({"register": 7, "f0_span": 1, "energy": 1, "quality": 1})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertAlmostEqual(weights["register"], 0.7)

    def test_invalid_percentile_is_rejected(self) -> None:
        source = {
            "register_percentile": 1.2,
            "f0_span_semitones": 12.0,
            "energy_percentile": 0.5,
        }
        prompt = {
            "prompt_id": "P0",
            "register_percentile": 0.5,
            "f0_span_semitones": 12.0,
            "energy_percentile": 0.5,
            "quality_score": 0.9,
        }
        with self.assertRaises(ValueError):
            rank_prompts(source, [prompt])


if __name__ == "__main__":
    unittest.main()
