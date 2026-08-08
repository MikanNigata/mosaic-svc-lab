import math
import unittest
from pathlib import Path

from mosaic_lab.temporal_features import TemporalKey, TemporalQuality, TemporalValueSummary
from mosaic_lab.temporal_memory import TemporalPatch
from mosaic_lab.temporal_retrieval import (
    GreedyTemporalPathSelector,
    RetrievalCandidate,
    RetrievalConfig,
    TemporalQueryFrame,
    rank_temporal_candidates,
    retrieval_confidence,
    retrieval_gate,
)


def _key(register: float, *, valid: bool = True) -> TemporalKey:
    return TemporalKey(register, 440.0, 430.0, 450.0, 0.8, 0.0, valid, 0.5, -20.0, 0.95, 0.9)


def _patch(patch_id: str, register: float, quality: float, centroid: float, start: float) -> TemporalPatch:
    return TemporalPatch(
        patch_id=patch_id,
        audio_path=Path(f"{patch_id}.wav"),
        feature_path=Path(f"{patch_id}.npz"),
        start_seconds=start,
        end_seconds=start + 0.4,
        key=_key(register),
        value_summary=TemporalValueSummary(centroid, 1500.0, 4000.0, 0.02, 0.8),
        quality=TemporalQuality(1.0, 0.0, 0.0, quality),
        accepted=True,
        rejection_reasons=(),
    )


class TemporalRetrievalTests(unittest.TestCase):
    def test_near_register_wins_and_quality_breaks_tie(self) -> None:
        query = TemporalQueryFrame(0, 0.0, _key(0.8))
        patches = [
            _patch("far", 0.2, 1.0, 1000.0, 0.0),
            _patch("near_bad", 0.8, 0.1, 1000.0, 0.1),
            _patch("near_good", 0.8, 0.9, 1000.0, 0.2),
        ]
        ranked = rank_temporal_candidates(query, patches, config=RetrievalConfig(top_k=3))
        self.assertEqual(ranked[0].patch_id, "near_good")
        self.assertEqual(len(ranked), 3)

    def test_missing_f0_features_do_not_fail(self) -> None:
        query = TemporalQueryFrame(0, 0.0, _key(0.5, valid=False))
        ranked = rank_temporal_candidates(
            query,
            [_patch("patch", 0.5, 0.9, 1000.0, 0.0)],
            config=RetrievalConfig(),
        )
        self.assertEqual(ranked[0].patch_id, "patch")
        self.assertTrue(math.isfinite(ranked[0].feature_distance))
        confidence = retrieval_confidence(query, ranked)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_continuity_weight_can_prevent_a_switch(self) -> None:
        patches = {
            "a": _patch("a", 0.4, 0.9, 1000.0, 0.0),
            "b": _patch("b", 0.6, 0.9, 7000.0, 10.0),
        }
        queries = [TemporalQueryFrame(0, 0.0, _key(0.4)), TemporalQueryFrame(1, 0.1, _key(0.6))]

        def candidate(patch_id: str, cost: float) -> RetrievalCandidate:
            patch = patches[patch_id]
            return RetrievalCandidate(patch_id, cost, 0.0, 0.0, cost, 0.5, 0.8, 0.9, patch.start_seconds, patch.key)

        candidates = [[candidate("a", 0.0), candidate("b", 0.4)], [candidate("a", 0.08), candidate("b", 0.0)]]
        selector = GreedyTemporalPathSelector(
            patches,
            continuity_weight=1.0,
            jump_penalty=0.2,
            expected_hop_seconds=0.1,
            temperature=0.15,
        )
        selected = selector.select(queries, candidates)
        self.assertEqual([item.patch_id for item in selected], ["a", "a"])

    def test_empty_memory_returns_no_candidates(self) -> None:
        query = TemporalQueryFrame(0, 0.0, _key(0.5))
        self.assertEqual(rank_temporal_candidates(query, [], config=RetrievalConfig()), [])
        self.assertEqual(retrieval_confidence(query, []), 0.0)

    def test_gate_accepts_unambiguous_matching_candidate(self) -> None:
        query = TemporalQueryFrame(0, 0.0, _key(0.5))
        candidates = rank_temporal_candidates(
            query,
            [
                _patch("near", 0.5, 0.98, 1000.0, 0.0),
                _patch("far", 0.9, 0.98, 1000.0, 1.0),
            ],
            config=RetrievalConfig(top_k=2, temperature=0.05),
        )
        confidence = retrieval_confidence(query, candidates)
        accepted, reasons, metrics = retrieval_gate(query, candidates, candidates[0], confidence)
        self.assertTrue(accepted)
        self.assertEqual(reasons, [])
        self.assertGreaterEqual(metrics["weight_margin"], 0.015)

    def test_gate_rejects_low_f0_confidence_and_register_mismatch(self) -> None:
        source = TemporalKey(0.1, 440.0, 430.0, 450.0, 0.8, 0.0, True, 0.5, -20.0, 0.95, 0.2)
        query = TemporalQueryFrame(0, 0.0, source)
        patch = _patch("high", 0.8, 0.98, 1000.0, 0.0)
        candidate = RetrievalCandidate(
            patch.patch_id,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.98,
            0.98,
            patch.start_seconds,
            patch.key,
        )
        accepted, reasons, _ = retrieval_gate(query, [candidate], candidate, 0.95)
        self.assertFalse(accepted)
        self.assertIn("source_f0_confidence", reasons)
        self.assertIn("register_distance", reasons)


if __name__ == "__main__":
    unittest.main()
