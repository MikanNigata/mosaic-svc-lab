from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from .temporal_features import TemporalKey, analyze_temporal_audio, extract_patch_features, iter_patch_bounds, json_safe
from .temporal_memory import TemporalPatch, load_memory_metadata, load_temporal_patches


@dataclass(frozen=True)
class TemporalQueryFrame:
    frame_index: int
    time_seconds: float
    key: TemporalKey


@dataclass(frozen=True)
class RetrievalCandidate:
    patch_id: str
    feature_distance: float
    continuity_cost: float
    jump_cost: float
    total_cost: float
    soft_weight: float
    confidence: float
    quality_score: float
    start_seconds: float
    key: TemporalKey


@dataclass(frozen=True)
class SelectedCandidate:
    frame_index: int
    patch_id: str | None
    candidate: RetrievalCandidate | None


class TemporalPathSelector(Protocol):
    def select(
        self,
        queries: list[TemporalQueryFrame],
        candidates: list[list[RetrievalCandidate]],
    ) -> list[SelectedCandidate]:
        ...


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    temperature: float = 0.15
    continuity_weight: float = 0.25
    jump_penalty: float = 0.05
    min_confidence: float = 0.0
    register_weight: float = 0.35
    f0_span_weight: float = 0.20
    f0_slope_weight: float = 0.15
    energy_weight: float = 0.15
    voiced_weight: float = 0.05
    quality_weight: float = 0.10
    confidence_nearest_weight: float = 0.35
    confidence_margin_weight: float = 0.20
    confidence_entropy_weight: float = 0.20
    confidence_valid_weight: float = 0.10
    confidence_quality_weight: float = 0.15

    def validate(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.continuity_weight < 0 or self.jump_penalty < 0:
            raise ValueError("continuity_weight and jump_penalty must be non-negative")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        weights = (
            self.register_weight,
            self.f0_span_weight,
            self.f0_slope_weight,
            self.energy_weight,
            self.voiced_weight,
            self.quality_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("retrieval weights must be non-negative with a positive sum")
        confidence_weights = (
            self.confidence_nearest_weight,
            self.confidence_margin_weight,
            self.confidence_entropy_weight,
            self.confidence_valid_weight,
            self.confidence_quality_weight,
        )
        if any(weight < 0 for weight in confidence_weights) or sum(confidence_weights) <= 0:
            raise ValueError("confidence weights must be non-negative with a positive sum")


def _valid(value: float, *, allow_zero: bool = True) -> bool:
    return math.isfinite(float(value)) and (allow_zero or float(value) > 0)


def _feature_distance(query: TemporalKey, patch: TemporalPatch, config: RetrievalConfig) -> float:
    components: list[tuple[float, float]] = []
    if _valid(query.relative_register) and _valid(patch.key.relative_register):
        components.append((config.register_weight, abs(query.relative_register - patch.key.relative_register)))
    if query.f0_valid and patch.key.f0_valid:
        components.append(
            (
                config.f0_span_weight,
                min(1.0, abs(query.f0_span_semitones - patch.key.f0_span_semitones) / 24.0),
            )
        )
        components.append(
            (
                config.f0_slope_weight,
                min(
                    1.0,
                    abs(
                        query.f0_slope_semitones_per_second
                        - patch.key.f0_slope_semitones_per_second
                    )
                    / 24.0,
                ),
            )
        )
    if _valid(query.energy_percentile) and _valid(patch.key.energy_percentile):
        components.append((config.energy_weight, abs(query.energy_percentile - patch.key.energy_percentile)))
    if _valid(query.voiced_ratio) and _valid(patch.key.voiced_ratio):
        components.append((config.voiced_weight, abs(query.voiced_ratio - patch.key.voiced_ratio)))
    if config.quality_weight > 0:
        components.append((config.quality_weight, -patch.quality.quality_score))
    total_weight = sum(weight for weight, _ in components)
    if total_weight <= 0:
        return math.inf
    return sum(weight * distance for weight, distance in components) / total_weight


def _softmax_weights(costs: list[float], temperature: float) -> list[float]:
    if not costs:
        return []
    finite_costs = [cost if math.isfinite(cost) else 1e6 for cost in costs]
    minimum = min(finite_costs)
    logits = [-(cost - minimum) / temperature for cost in finite_costs]
    exponentials = [math.exp(max(-700.0, min(700.0, value))) for value in logits]
    denominator = sum(exponentials)
    if denominator <= 0:
        return [1.0 / len(costs)] * len(costs)
    return [value / denominator for value in exponentials]


def _value_transition(left: TemporalPatch, right: TemporalPatch) -> float:
    a, b = left.value_summary, right.value_summary
    distances = (
        abs(a.spectral_centroid_hz - b.spectral_centroid_hz) / 8000.0,
        abs(a.spectral_bandwidth_hz - b.spectral_bandwidth_hz) / 8000.0,
        abs(a.spectral_rolloff_hz - b.spectral_rolloff_hz) / 10000.0,
        abs(a.spectral_flatness - b.spectral_flatness) / 0.20,
        abs(a.harmonic_ratio - b.harmonic_ratio),
    )
    return sum(min(1.0, value) for value in distances) / len(distances)


class GreedyTemporalPathSelector:
    """Greedy P0 selector with an interface that can later be replaced by Viterbi."""

    def __init__(
        self,
        patches: dict[str, TemporalPatch],
        *,
        continuity_weight: float,
        jump_penalty: float,
        expected_hop_seconds: float,
        temperature: float,
    ) -> None:
        self.patches = patches
        self.continuity_weight = continuity_weight
        self.jump_penalty = jump_penalty
        self.expected_hop_seconds = expected_hop_seconds
        self.temperature = temperature

    def select(
        self,
        queries: list[TemporalQueryFrame],
        candidates: list[list[RetrievalCandidate]],
    ) -> list[SelectedCandidate]:
        selected: list[SelectedCandidate] = []
        previous: RetrievalCandidate | None = None
        for query, frame_candidates in zip(queries, candidates):
            scored: list[RetrievalCandidate] = []
            for candidate in frame_candidates:
                continuity = 0.0
                jump = 0.0
                if previous is not None:
                    previous_patch = self.patches[previous.patch_id]
                    current_patch = self.patches[candidate.patch_id]
                    continuity = _value_transition(previous_patch, current_patch)
                    expected = previous_patch.start_seconds + self.expected_hop_seconds
                    if abs(current_patch.start_seconds - expected) > max(0.5, 3.0 * self.expected_hop_seconds):
                        jump = self.jump_penalty
                total = candidate.feature_distance + self.continuity_weight * continuity + jump
                scored.append(
                    replace(
                        candidate,
                        continuity_cost=continuity,
                        jump_cost=jump,
                        total_cost=total,
                    )
                )
            scored.sort(key=lambda item: (item.total_cost, item.patch_id))
            weights = _softmax_weights([item.total_cost for item in scored], self.temperature)
            scored = [replace(item, soft_weight=weight) for item, weight in zip(scored, weights)]
            chosen = scored[0] if scored else None
            selected.append(
                SelectedCandidate(
                    frame_index=query.frame_index,
                    patch_id=chosen.patch_id if chosen else None,
                    candidate=chosen,
                )
            )
            previous = chosen
            frame_candidates[:] = scored
        return selected


def rank_temporal_candidates(
    query: TemporalQueryFrame,
    patches: list[TemporalPatch],
    *,
    config: RetrievalConfig,
) -> list[RetrievalCandidate]:
    """Return the feature-distance top-k candidates for one query frame."""
    scored: list[RetrievalCandidate] = []
    for patch in patches:
        distance = _feature_distance(query.key, patch, config)
        if not math.isfinite(distance):
            continue
        scored.append(
            RetrievalCandidate(
                patch_id=patch.patch_id,
                feature_distance=distance,
                continuity_cost=0.0,
                jump_cost=0.0,
                total_cost=distance,
                soft_weight=0.0,
                confidence=math.exp(-3.0 * max(0.0, distance)) * patch.quality.quality_score,
                quality_score=patch.quality.quality_score,
                start_seconds=patch.start_seconds,
                key=patch.key,
            )
        )
    scored.sort(key=lambda item: (item.feature_distance, item.patch_id))
    top = scored[: min(config.top_k, len(scored))]
    weights = _softmax_weights([item.feature_distance for item in top], config.temperature)
    return [replace(item, soft_weight=weight) for item, weight in zip(top, weights)]


def retrieval_confidence(
    query: TemporalQueryFrame,
    candidates: list[RetrievalCandidate],
    config: RetrievalConfig | None = None,
) -> float:
    """Combine proximity, margin, entropy, valid features, and patch quality."""
    if not candidates:
        return 0.0
    config = config or RetrievalConfig()
    ordered = sorted(candidates, key=lambda item: item.total_cost)
    first = ordered[0]
    nearest = math.exp(-3.0 * max(0.0, first.feature_distance))
    if len(ordered) > 1:
        margin = max(0.0, ordered[1].feature_distance - first.feature_distance)
        margin_score = min(1.0, margin / max(0.10, abs(ordered[1].feature_distance)))
    else:
        margin_score = 1.0
    weights = [max(0.0, item.soft_weight) for item in ordered]
    entropy = -sum(weight * math.log(max(weight, 1e-12)) for weight in weights)
    entropy_score = 1.0 if len(weights) == 1 else max(0.0, 1.0 - entropy / math.log(len(weights)))
    valid_features = 3 + (2 if query.key.f0_valid else 0)
    valid_score = valid_features / 5.0
    weighted = (
        (config.confidence_nearest_weight, nearest),
        (config.confidence_margin_weight, margin_score),
        (config.confidence_entropy_weight, entropy_score),
        (config.confidence_valid_weight, valid_score),
        (config.confidence_quality_weight, first.quality_score),
    )
    confidence = sum(weight * value for weight, value in weighted) / sum(weight for weight, _ in weighted)
    return max(0.0, min(1.0, confidence))


def build_temporal_queries(
    source: str | Path,
    *,
    patch_seconds: float,
    update_seconds: float,
    analysis_sr: int,
) -> list[TemporalQueryFrame]:
    analysis = analyze_temporal_audio(source, analysis_sr=analysis_sr)
    queries: list[TemporalQueryFrame] = []
    for frame_index, (start, end, start_seconds, _) in enumerate(
        iter_patch_bounds(analysis, patch_seconds=patch_seconds, hop_seconds=update_seconds)
    ):
        features = extract_patch_features(analysis, start_sample=start, end_sample=end)
        queries.append(
            TemporalQueryFrame(
                frame_index=frame_index,
                time_seconds=start_seconds,
                key=features.key,
            )
        )
    if not queries:
        raise ValueError(
            f"Temporal query source is shorter than the memory patch length ({patch_seconds:.3f}s)"
        )
    return queries


def query_temporal_memory(
    source: str | Path,
    memory: str | Path,
    output: str | Path,
    *,
    config: RetrievalConfig | None = None,
    update_seconds: float = 0.10,
    disable_smoothing: bool = False,
) -> Path:
    """Query a Temporal Timbre Memory and write an inspectable JSONL path."""
    config = config or RetrievalConfig()
    config.validate()
    if update_seconds <= 0:
        raise ValueError("update_seconds must be positive")
    metadata = load_memory_metadata(memory)
    analysis_config = metadata["analysis"]
    patches = load_temporal_patches(memory, accepted_only=True)
    queries = build_temporal_queries(
        source,
        patch_seconds=float(analysis_config["patch_seconds"]),
        update_seconds=update_seconds,
        analysis_sr=int(analysis_config["analysis_sr"]),
    )
    frame_candidates = [rank_temporal_candidates(query, patches, config=config) for query in queries]
    selector = GreedyTemporalPathSelector(
        {patch.patch_id: patch for patch in patches},
        continuity_weight=0.0 if disable_smoothing else config.continuity_weight,
        jump_penalty=0.0 if disable_smoothing else config.jump_penalty,
        expected_hop_seconds=float(analysis_config["hop_seconds"]),
        temperature=config.temperature,
    )
    selections = selector.select(queries, frame_candidates)
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for query, candidates, selection in zip(queries, frame_candidates, selections):
            confidence = retrieval_confidence(query, candidates, config)
            selected_id = selection.patch_id if confidence >= config.min_confidence else None
            record = json_safe(
                {
                    "schema_version": 1,
                    "frame_index": query.frame_index,
                    "source_time_seconds": query.time_seconds,
                    "source_features": asdict(query.key),
                    "candidates": [
                        {
                            "patch_id": item.patch_id,
                            "score": min(1.0, max(0.0, 1.0 - item.total_cost)),
                            "feature_distance": item.feature_distance,
                            "continuity_cost": item.continuity_cost,
                            "jump_cost": item.jump_cost,
                            "total_cost": item.total_cost,
                            "soft_weight": item.soft_weight,
                            "confidence": item.confidence,
                            "target_features": asdict(item.key),
                        }
                        for item in candidates
                    ],
                    "selected_patch_id": selected_id,
                    "retrieval_confidence": confidence if selected_id else 0.0,
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    summary = summarize_query_records(load_query_records(destination))
    summary_path = destination.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_query_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Temporal query result does not exist: {source}")
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
    return records


def summarize_query_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [record.get("selected_patch_id") for record in records]
    selected_records = [record for record in records if record.get("selected_patch_id")]
    confidences = sorted(float(record.get("retrieval_confidence", 0.0)) for record in records)
    distances = []
    switches = 0
    previous: str | None = None
    coverage_counts = {"low": [0, 0], "mid": [0, 0], "high": [0, 0]}
    for record in records:
        patch_id = record.get("selected_patch_id")
        if patch_id and previous and patch_id != previous:
            switches += 1
        if patch_id:
            previous = patch_id
            candidate = next(
                (item for item in record.get("candidates", []) if item.get("patch_id") == patch_id),
                None,
            )
            if candidate:
                distances.append(float(candidate.get("feature_distance", 0.0)))
        register = float(record.get("source_features", {}).get("relative_register", 0.5))
        band = "low" if register < 1.0 / 3.0 else "high" if register >= 2.0 / 3.0 else "mid"
        coverage_counts[band][1] += 1
        coverage_counts[band][0] += int(bool(patch_id))
    duration = 0.0
    if len(records) > 1:
        duration = max(0.0, float(records[-1]["source_time_seconds"]) - float(records[0]["source_time_seconds"]))
    median_confidence = 0.0
    if confidences:
        middle = len(confidences) // 2
        median_confidence = confidences[middle] if len(confidences) % 2 else (confidences[middle - 1] + confidences[middle]) / 2.0
    return json_safe(
        {
            "query_frames": len(records),
            "frames_with_selection": len(selected_records),
            "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "median_confidence": median_confidence,
            "patch_switches": switches,
            "switches_per_second": switches / duration if duration > 0 else 0.0,
            "unique_patches_used": len({item for item in selected if item}),
            "mean_feature_distance": sum(distances) / len(distances) if distances else 0.0,
            "low_confidence_ratio": sum(value < 0.5 for value in confidences) / len(confidences) if confidences else 0.0,
            "register_coverage": {
                band: selected_count / total if total else 0.0
                for band, (selected_count, total) in coverage_counts.items()
            },
        }
    )
