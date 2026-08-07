from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


FEATURE_VERSION = 1
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 256


def _audio_modules():
    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            'Temporal audio features require the optional audio dependencies.\n'
            'Install with:\n\npython -m pip install -e ".[audio]"'
        ) from exc
    return librosa, np, sf


def finite_float(value: Any, default: float = 0.0) -> float:
    """Return a JSON-safe finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite numeric values before JSON serialization."""
    if isinstance(value, float):
        return finite_float(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class TemporalKey:
    relative_register: float
    f0_median_hz: float
    f0_p10_hz: float
    f0_p90_hz: float
    f0_span_semitones: float
    f0_slope_semitones_per_second: float
    f0_valid: bool
    energy_percentile: float
    rms_db: float
    voiced_ratio: float
    f0_confidence: float


@dataclass(frozen=True)
class TemporalValueSummary:
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_rolloff_hz: float
    spectral_flatness: float
    harmonic_ratio: float


@dataclass(frozen=True)
class TemporalQuality:
    active_ratio: float
    clipping_ratio: float
    nonfinite_ratio: float
    quality_score: float


@dataclass(frozen=True)
class SpeakerStatistics:
    log_f0_median: float
    log_f0_p05: float
    log_f0_p95: float
    rms_db_p05: float
    rms_db_p95: float


@dataclass(frozen=True)
class TemporalPatchFeatures:
    key: TemporalKey
    value_summary: TemporalValueSummary
    quality: TemporalQuality
    arrays: dict[str, Any]


@dataclass
class TemporalAnalysis:
    waveform: Any
    sample_rate: int
    frame_times: Any
    f0_hz: Any
    voiced_flag: Any
    voiced_probability: Any
    rms_db: Any
    mel_db: Any
    mfcc: Any
    spectral_centroid: Any
    spectral_bandwidth: Any
    spectral_rolloff: Any
    spectral_flatness: Any
    speaker_statistics: SpeakerStatistics
    global_peak: float
    source_path: Path

    @property
    def duration_seconds(self) -> float:
        return float(self.waveform.size / self.sample_rate)


def _quantile(values: Any, q: float, *, default: float = 0.0) -> float:
    _, np, _ = _audio_modules()
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return default
    return finite_float(np.quantile(finite, q), default)


def _percentile_rank(value: float, distribution: Any, low: float, high: float) -> float:
    _, np, _ = _audio_modules()
    values = np.asarray(distribution, dtype=float)
    values = values[np.isfinite(values)]
    if not math.isfinite(value) or values.size == 0:
        return 0.5
    clipped = np.clip(values, low, high)
    target = float(np.clip(value, low, high))
    return finite_float(np.mean(clipped <= target), 0.5)


def analyze_temporal_audio(
    path: str | Path,
    *,
    analysis_sr: int = 22050,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> TemporalAnalysis:
    """Extract reusable frame-level features for temporal patch analysis."""
    librosa, np, _ = _audio_modules()
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Temporal audio source does not exist: {source}")
    if analysis_sr <= 0:
        raise ValueError("analysis_sr must be positive")
    try:
        waveform, _ = librosa.load(str(source), sr=analysis_sr, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode temporal audio '{source}': {exc}") from exc
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"Temporal audio is empty: {source}")
    nonfinite = ~np.isfinite(waveform)
    if np.any(nonfinite):
        waveform = waveform.copy()
        waveform[nonfinite] = 0.0

    f0_hz, voiced_flag, voiced_probability = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=analysis_sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=1.0)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=analysis_sr,
        n_fft=frame_length,
        hop_length=hop_length,
        n_mels=64,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(
        y=waveform, sr=analysis_sr, n_fft=frame_length, hop_length=hop_length
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=waveform, sr=analysis_sr, n_fft=frame_length, hop_length=hop_length
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=waveform, sr=analysis_sr, n_fft=frame_length, hop_length=hop_length, roll_percent=0.85
    )[0]
    flatness = librosa.feature.spectral_flatness(
        y=waveform, n_fft=frame_length, hop_length=hop_length
    )[0]
    frame_count = min(
        len(f0_hz),
        len(rms_db),
        mel_db.shape[1],
        mfcc.shape[1],
        len(centroid),
        len(bandwidth),
        len(rolloff),
        len(flatness),
    )
    if frame_count == 0:
        raise ValueError(f"Temporal audio produced no analysis frames: {source}")
    f0_hz = np.asarray(f0_hz[:frame_count], dtype=np.float32)
    voiced_flag = np.asarray(voiced_flag[:frame_count], dtype=bool)
    voiced_probability = np.asarray(voiced_probability[:frame_count], dtype=np.float32)
    rms_db = np.asarray(rms_db[:frame_count], dtype=np.float32)
    mel_db = np.asarray(mel_db[:, :frame_count], dtype=np.float32)
    mfcc = np.asarray(mfcc[:, :frame_count], dtype=np.float32)
    centroid = np.asarray(centroid[:frame_count], dtype=np.float32)
    bandwidth = np.asarray(bandwidth[:frame_count], dtype=np.float32)
    rolloff = np.asarray(rolloff[:frame_count], dtype=np.float32)
    flatness = np.asarray(flatness[:frame_count], dtype=np.float32)
    frame_times = librosa.frames_to_time(
        np.arange(frame_count), sr=analysis_sr, hop_length=hop_length
    ).astype(np.float32)

    valid_f0 = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    log_f0 = np.log(valid_f0) if valid_f0.size else np.asarray([], dtype=float)
    stats = SpeakerStatistics(
        log_f0_median=_quantile(log_f0, 0.50),
        log_f0_p05=_quantile(log_f0, 0.05),
        log_f0_p95=_quantile(log_f0, 0.95),
        rms_db_p05=_quantile(rms_db, 0.05, default=-80.0),
        rms_db_p95=_quantile(rms_db, 0.95, default=-20.0),
    )
    return TemporalAnalysis(
        waveform=waveform,
        sample_rate=analysis_sr,
        frame_times=frame_times,
        f0_hz=f0_hz,
        voiced_flag=voiced_flag,
        voiced_probability=voiced_probability,
        rms_db=rms_db,
        mel_db=mel_db,
        mfcc=mfcc,
        spectral_centroid=centroid,
        spectral_bandwidth=bandwidth,
        spectral_rolloff=rolloff,
        spectral_flatness=flatness,
        speaker_statistics=stats,
        global_peak=finite_float(np.max(np.abs(waveform))),
        source_path=source,
    )


def iter_patch_bounds(
    analysis: TemporalAnalysis,
    *,
    patch_seconds: float,
    hop_seconds: float,
) -> Iterator[tuple[int, int, float, float]]:
    """Yield deterministic full-length patch sample bounds."""
    if patch_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("patch_seconds and hop_seconds must be positive")
    patch_samples = max(1, round(patch_seconds * analysis.sample_rate))
    hop_samples = max(1, round(hop_seconds * analysis.sample_rate))
    if analysis.waveform.size < patch_samples:
        return
    index = 0
    while index + patch_samples <= analysis.waveform.size:
        yield (
            index,
            index + patch_samples,
            index / analysis.sample_rate,
            (index + patch_samples) / analysis.sample_rate,
        )
        index += hop_samples


def _harmonic_ratio(segment: Any, median_f0: float, sample_rate: int) -> float:
    _, np, _ = _audio_modules()
    if not math.isfinite(median_f0) or median_f0 <= 0 or segment.size < 4:
        return 0.0
    lag = int(round(sample_rate / median_f0))
    if lag <= 0 or lag >= segment.size // 2:
        return 0.0
    centered = segment.astype(float) - float(np.mean(segment))
    numerator = float(np.dot(centered[:-lag], centered[lag:]))
    denominator = float(np.sqrt(np.dot(centered[:-lag], centered[:-lag]) * np.dot(centered[lag:], centered[lag:])))
    return float(np.clip(numerator / max(denominator, 1e-12), 0.0, 1.0))


def extract_patch_features(
    analysis: TemporalAnalysis,
    *,
    start_sample: int,
    end_sample: int,
) -> TemporalPatchFeatures:
    """Aggregate one patch from precomputed frame-level analysis."""
    _, np, _ = _audio_modules()
    if start_sample < 0 or end_sample <= start_sample or end_sample > analysis.waveform.size:
        raise ValueError("invalid temporal patch bounds")
    start_seconds = start_sample / analysis.sample_rate
    end_seconds = end_sample / analysis.sample_rate
    frame_mask = (analysis.frame_times >= start_seconds) & (analysis.frame_times < end_seconds)
    frame_indices = np.flatnonzero(frame_mask)
    if frame_indices.size == 0:
        nearest = int(np.argmin(np.abs(analysis.frame_times - (start_seconds + end_seconds) / 2.0)))
        frame_indices = np.asarray([nearest])

    segment = np.asarray(analysis.waveform[start_sample:end_sample], dtype=np.float32)
    patch_f0 = analysis.f0_hz[frame_indices]
    patch_voiced = np.isfinite(patch_f0) & (patch_f0 > 0)
    voiced_f0 = patch_f0[patch_voiced]
    probabilities = analysis.voiced_probability[frame_indices]
    if voiced_f0.size:
        f0_median = finite_float(np.median(voiced_f0))
        f0_p10 = finite_float(np.quantile(voiced_f0, 0.10))
        f0_p90 = finite_float(np.quantile(voiced_f0, 0.90))
        f0_span = finite_float(12.0 * np.log2(max(f0_p90, 1e-6) / max(f0_p10, 1e-6)))
    else:
        f0_median = f0_p10 = f0_p90 = f0_span = 0.0
    f0_valid = bool(voiced_f0.size >= 2)
    slope = 0.0
    if f0_valid:
        voiced_times = analysis.frame_times[frame_indices][patch_voiced]
        relative_times = voiced_times - voiced_times[0]
        if float(np.ptp(relative_times)) > 0:
            semitones = 12.0 * np.log2(voiced_f0)
            slope = finite_float(np.polyfit(relative_times, semitones, 1)[0])

    stats = analysis.speaker_statistics
    log_f0_distribution = np.log(
        analysis.f0_hz[np.isfinite(analysis.f0_hz) & (analysis.f0_hz > 0)]
    )
    relative_register = _percentile_rank(
        math.log(f0_median) if f0_median > 0 else math.nan,
        log_f0_distribution,
        stats.log_f0_p05,
        stats.log_f0_p95,
    )
    patch_rms_db = finite_float(np.mean(analysis.rms_db[frame_indices]), -100.0)
    energy_percentile = _percentile_rank(
        patch_rms_db,
        analysis.rms_db,
        stats.rms_db_p05,
        stats.rms_db_p95,
    )
    absolute = np.abs(segment)
    active_threshold = max(1e-5, analysis.global_peak * (10.0 ** (-45.0 / 20.0)))
    active_ratio = finite_float(np.mean(absolute > active_threshold))
    clipping_ratio = finite_float(np.mean(absolute >= 0.999))
    nonfinite_ratio = finite_float(np.mean(~np.isfinite(segment)))
    f0_confidence = finite_float(np.mean(probabilities[patch_voiced])) if np.any(patch_voiced) else 0.0
    voiced_ratio = finite_float(np.mean(patch_voiced))
    quality_score = float(
        np.clip(
            1.0
            - 0.30 * min(1.0, clipping_ratio / 0.001)
            - 0.20 * max(0.0, 0.5 - active_ratio) / 0.5
            - 0.25 * max(0.0, 0.7 - f0_confidence) / 0.7
            - 0.15 * min(1.0, finite_float(np.mean(analysis.spectral_flatness[frame_indices])) / 0.12)
            - 0.10 * min(1.0, nonfinite_ratio / 0.01),
            0.0,
            1.0,
        )
    )
    key = TemporalKey(
        relative_register=relative_register,
        f0_median_hz=f0_median,
        f0_p10_hz=f0_p10,
        f0_p90_hz=f0_p90,
        f0_span_semitones=f0_span,
        f0_slope_semitones_per_second=slope,
        f0_valid=f0_valid,
        energy_percentile=energy_percentile,
        rms_db=patch_rms_db,
        voiced_ratio=voiced_ratio,
        f0_confidence=f0_confidence,
    )
    summary = TemporalValueSummary(
        spectral_centroid_hz=finite_float(np.mean(analysis.spectral_centroid[frame_indices])),
        spectral_bandwidth_hz=finite_float(np.mean(analysis.spectral_bandwidth[frame_indices])),
        spectral_rolloff_hz=finite_float(np.mean(analysis.spectral_rolloff[frame_indices])),
        spectral_flatness=finite_float(np.mean(analysis.spectral_flatness[frame_indices])),
        harmonic_ratio=_harmonic_ratio(segment, f0_median, analysis.sample_rate),
    )
    quality = TemporalQuality(
        active_ratio=active_ratio,
        clipping_ratio=clipping_ratio,
        nonfinite_ratio=nonfinite_ratio,
        quality_score=quality_score,
    )
    arrays = {
        "frame_times_seconds": analysis.frame_times[frame_indices] - start_seconds,
        "f0_hz": patch_f0,
        "voiced_probability": probabilities,
        "rms_db": analysis.rms_db[frame_indices],
        "log_mel": analysis.mel_db[:, frame_indices],
        "log_mel_mean": np.mean(analysis.mel_db[:, frame_indices], axis=1),
        "log_mel_std": np.std(analysis.mel_db[:, frame_indices], axis=1),
        "mfcc": analysis.mfcc[:, frame_indices],
        "mfcc_mean": np.mean(analysis.mfcc[:, frame_indices], axis=1),
        "mfcc_std": np.std(analysis.mfcc[:, frame_indices], axis=1),
        "spectral_centroid_hz": analysis.spectral_centroid[frame_indices],
        "spectral_bandwidth_hz": analysis.spectral_bandwidth[frame_indices],
        "spectral_rolloff_hz": analysis.spectral_rolloff[frame_indices],
        "spectral_flatness": analysis.spectral_flatness[frame_indices],
    }
    arrays = {name: np.nan_to_num(np.asarray(value), nan=0.0, posinf=0.0, neginf=0.0) for name, value in arrays.items()}
    return TemporalPatchFeatures(key=key, value_summary=summary, quality=quality, arrays=arrays)


def feature_record(features: TemporalPatchFeatures) -> dict[str, Any]:
    """Convert patch features to a compact JSON-safe record."""
    return json_safe(
        {
            "key": asdict(features.key),
            "value_summary": asdict(features.value_summary),
            "quality": asdict(features.quality),
            "phonetic": {"type": "none", "feature_path": None},
        }
    )
