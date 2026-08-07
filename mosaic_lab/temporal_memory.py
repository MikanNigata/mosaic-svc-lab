from __future__ import annotations

import hashlib
import json
import platform
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .temporal_features import (
    FEATURE_VERSION,
    TemporalKey,
    TemporalQuality,
    TemporalValueSummary,
    _audio_modules,
    analyze_temporal_audio,
    extract_patch_features,
    feature_record,
    iter_patch_bounds,
    json_safe,
)


SCHEMA_VERSION = 1
MEMORY_TYPE = "mosaic_temporal_timbre_memory"


@dataclass(frozen=True)
class TemporalPatch:
    patch_id: str
    audio_path: Path
    feature_path: Path
    start_seconds: float
    end_seconds: float
    key: TemporalKey
    value_summary: TemporalValueSummary
    quality: TemporalQuality
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class EnrollmentConfig:
    patch_seconds: float = 0.40
    hop_seconds: float = 0.10
    analysis_sr: int = 22050
    min_active_ratio: float = 0.50
    min_f0_confidence: float = 0.50
    max_clipping_ratio: float = 0.001

    def validate(self) -> None:
        if self.patch_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError("patch_seconds and hop_seconds must be positive")
        if self.analysis_sr <= 0:
            raise ValueError("analysis_sr must be positive")
        for name in ("min_active_ratio", "min_f0_confidence", "max_clipping_ratio"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    librosa, np, sf = _audio_modules()
    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "librosa": str(librosa.__version__),
        "soundfile": str(sf.__version__),
    }


def _rejection_reasons(features: Any, config: EnrollmentConfig) -> tuple[str, ...]:
    reasons: list[str] = []
    if features.quality.active_ratio < config.min_active_ratio:
        reasons.append("low_active_ratio")
    if features.quality.clipping_ratio > config.max_clipping_ratio:
        reasons.append("high_clipping_ratio")
    if features.quality.nonfinite_ratio > 0.0:
        reasons.append("nonfinite_audio")
    if features.key.rms_db < -65.0:
        reasons.append("extreme_silence")
    if features.key.f0_confidence < config.min_f0_confidence:
        reasons.append("low_f0_confidence")
    if not features.key.f0_valid:
        reasons.append("insufficient_voiced_frames")
    return tuple(reasons)


def _prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"Temporal memory output exists and is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Temporal memory output is not empty: {output}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "patches").mkdir()
    (output / "features").mkdir()


def build_temporal_memory(
    source: str | Path,
    output: str | Path,
    *,
    config: EnrollmentConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Build an inspectable Temporal Timbre Memory from target singing."""
    _, np, sf = _audio_modules()
    config = config or EnrollmentConfig()
    config.validate()
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Temporal enrollment source does not exist: {source_path}")
    _prepare_output(output_path, overwrite)
    analysis = analyze_temporal_audio(source_path, analysis_sr=config.analysis_sr)
    bounds = list(
        iter_patch_bounds(
            analysis,
            patch_seconds=config.patch_seconds,
            hop_seconds=config.hop_seconds,
        )
    )
    if not bounds:
        raise ValueError(
            f"Temporal enrollment source ({analysis.duration_seconds:.3f}s) is shorter "
            f"than patch_seconds ({config.patch_seconds:.3f}s)"
        )

    records: list[dict[str, Any]] = []
    accepted_count = 0
    for patch_number, (start_sample, end_sample, start_seconds, end_seconds) in enumerate(bounds, start=1):
        patch_id = f"patch_{patch_number:06d}"
        audio_relative = Path("patches") / f"{patch_id}.wav"
        feature_relative = Path("features") / f"{patch_id}.npz"
        rejection_reasons: tuple[str, ...]
        try:
            features = extract_patch_features(
                analysis,
                start_sample=start_sample,
                end_sample=end_sample,
            )
            rejection_reasons = _rejection_reasons(features, config)
            np.savez_compressed(output_path / feature_relative, **features.arrays)
            record_features = feature_record(features)
        except Exception as exc:
            rejection_reasons = ("analysis_failure",)
            record_features = {
                "key": asdict(
                    TemporalKey(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.5, -100.0, 0.0, 0.0)
                ),
                "value_summary": asdict(TemporalValueSummary(0.0, 0.0, 0.0, 0.0, 0.0)),
                "quality": asdict(TemporalQuality(0.0, 0.0, 0.0, 0.0)),
                "phonetic": {"type": "none", "feature_path": None},
                "analysis_error": f"{type(exc).__name__}: {exc}",
            }
            np.savez_compressed(output_path / feature_relative, analysis_error=np.asarray([str(exc)]))
        sf.write(
            str(output_path / audio_relative),
            analysis.waveform[start_sample:end_sample],
            analysis.sample_rate,
            subtype="PCM_16",
        )
        accepted = not rejection_reasons
        accepted_count += int(accepted)
        records.append(
            json_safe(
                {
                    "schema_version": SCHEMA_VERSION,
                    "patch_id": patch_id,
                    "audio_path": audio_relative.as_posix(),
                    "feature_path": feature_relative.as_posix(),
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "duration_seconds": end_seconds - start_seconds,
                    "accepted": accepted,
                    "rejection_reasons": list(rejection_reasons),
                    **record_features,
                    "extraction_settings": {
                        "analysis_sr": config.analysis_sr,
                        "patch_seconds": config.patch_seconds,
                        "hop_seconds": config.hop_seconds,
                        "feature_version": FEATURE_VERSION,
                    },
                }
            )
        )

    metadata = json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "memory_type": MEMORY_TYPE,
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "analysis": {
                "analysis_sr": config.analysis_sr,
                "patch_seconds": config.patch_seconds,
                "hop_seconds": config.hop_seconds,
                "feature_version": FEATURE_VERSION,
                "thresholds": {
                    "min_active_ratio": config.min_active_ratio,
                    "min_f0_confidence": config.min_f0_confidence,
                    "max_clipping_ratio": config.max_clipping_ratio,
                },
            },
            "speaker_statistics": asdict(analysis.speaker_statistics),
            "phonetic": {"type": "none", "feature_path": None},
            "patch_count": len(records),
            "accepted_patch_count": accepted_count,
            "rejected_patch_count": len(records) - accepted_count,
            "runtime_versions": _versions(),
        }
    )
    (output_path / "memory.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_path / "memory.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return output_path / "memory.json"


def load_memory_metadata(memory: str | Path) -> dict[str, Any]:
    root = Path(memory).resolve()
    metadata_path = root / "memory.json" if root.is_dir() else root
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Temporal memory metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("memory_type") != MEMORY_TYPE:
        raise ValueError(f"Unsupported temporal memory type in {metadata_path}")
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported temporal memory schema in {metadata_path}")
    return metadata


def iter_memory_records(memory: str | Path, *, accepted_only: bool = False) -> Iterator[dict[str, Any]]:
    root = Path(memory).resolve()
    if root.is_file():
        root = root.parent
    records_path = root / "memory.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(f"Temporal memory index does not exist: {records_path}")
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {records_path}:{line_number}: {exc}") from exc
            if not accepted_only or bool(record.get("accepted")):
                yield record


def load_temporal_patches(memory: str | Path, *, accepted_only: bool = True) -> list[TemporalPatch]:
    root = Path(memory).resolve()
    if root.is_file():
        root = root.parent
    patches: list[TemporalPatch] = []
    for record in iter_memory_records(root, accepted_only=accepted_only):
        key = TemporalKey(**record["key"])
        summary = TemporalValueSummary(**record["value_summary"])
        quality = TemporalQuality(**record["quality"])
        patches.append(
            TemporalPatch(
                patch_id=str(record["patch_id"]),
                audio_path=(root / record["audio_path"]).resolve(),
                feature_path=(root / record["feature_path"]).resolve(),
                start_seconds=float(record["start_seconds"]),
                end_seconds=float(record["end_seconds"]),
                key=key,
                value_summary=summary,
                quality=quality,
                accepted=bool(record["accepted"]),
                rejection_reasons=tuple(record.get("rejection_reasons", [])),
            )
        )
    return patches
