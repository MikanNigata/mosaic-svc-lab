from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_WEIGHTS: dict[str, float] = {
    "register": 0.35,
    "f0_span": 0.25,
    "energy": 0.20,
    "quality": 0.20,
}


@dataclass(frozen=True)
class RankedPrompt:
    prompt_id: str
    score: float
    components: dict[str, float]
    record: dict[str, Any]


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {number!r}")
    return number


def _unit_interval(value: Any, field: str) -> float:
    number = _finite_number(value, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be in [0, 1], got {number}")
    return number


def _bounded_similarity(a: float, b: float, scale: float) -> float:
    if scale <= 0:
        raise ValueError("similarity scale must be positive")
    return max(0.0, 1.0 - abs(a - b) / scale)


def _relative_similarity(a: float, b: float, floor: float = 1.0) -> float:
    denominator = max(abs(a), abs(b), floor)
    return max(0.0, 1.0 - abs(a - b) / denominator)


def validate_feature_record(record: Mapping[str, Any], *, source: bool = False) -> None:
    prefix = "source" if source else f"prompt {record.get('prompt_id', '<unknown>')}"
    _unit_interval(record.get("register_percentile"), f"{prefix}.register_percentile")
    _finite_number(record.get("f0_span_semitones"), f"{prefix}.f0_span_semitones")
    _unit_interval(record.get("energy_percentile"), f"{prefix}.energy_percentile")
    if not source:
        prompt_id = record.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError("prompt_id must be a non-empty string")
        _unit_interval(record.get("quality_score"), f"{prefix}.quality_score")


def normalize_weights(weights: Mapping[str, Any] | None = None) -> dict[str, float]:
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown retrieval weights: {sorted(unknown)}")
        merged.update({key: _finite_number(value, f"weight.{key}") for key, value in weights.items()})
    if any(value < 0 for value in merged.values()):
        raise ValueError("retrieval weights must be non-negative")
    total = sum(merged.values())
    if total <= 0:
        raise ValueError("at least one retrieval weight must be positive")
    return {key: value / total for key, value in merged.items()}


def score_prompt(
    source_features: Mapping[str, Any],
    prompt: Mapping[str, Any],
    *,
    weights: Mapping[str, Any] | None = None,
) -> RankedPrompt:
    validate_feature_record(source_features, source=True)
    validate_feature_record(prompt, source=False)
    normalized_weights = normalize_weights(weights)

    components = {
        "register": _bounded_similarity(
            float(source_features["register_percentile"]),
            float(prompt["register_percentile"]),
            1.0,
        ),
        "f0_span": _relative_similarity(
            float(source_features["f0_span_semitones"]),
            float(prompt["f0_span_semitones"]),
        ),
        "energy": _bounded_similarity(
            float(source_features["energy_percentile"]),
            float(prompt["energy_percentile"]),
            1.0,
        ),
        "quality": float(prompt["quality_score"]),
    }
    score = sum(normalized_weights[name] * components[name] for name in normalized_weights)
    return RankedPrompt(
        prompt_id=str(prompt["prompt_id"]),
        score=score,
        components=components,
        record=dict(prompt),
    )


def rank_prompts(
    source_features: Mapping[str, Any],
    prompts: Iterable[Mapping[str, Any]],
    *,
    weights: Mapping[str, Any] | None = None,
    top_k: int | None = None,
) -> list[RankedPrompt]:
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    ranked = [score_prompt(source_features, prompt, weights=weights) for prompt in prompts]
    ranked.sort(key=lambda item: (-item.score, item.prompt_id))
    return ranked if top_k is None else ranked[:top_k]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            records.append(record)
    return records


def dump_ranking(path: str | Path, ranking: Sequence[RankedPrompt]) -> None:
    output = [
        {
            "rank": index,
            "prompt_id": item.prompt_id,
            "score": round(item.score, 8),
            "components": {name: round(value, 8) for name, value in item.components.items()},
            "path": item.record.get("path"),
        }
        for index, item in enumerate(ranking, start=1)
    ]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
