from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable


def _audio_modules():
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("audio features require: python -m pip install -e .[audio]") from exc
    return librosa, np


def analyze_audio(path: str | Path, *, sr: int = 22050) -> dict[str, float]:
    librosa, np = _audio_modules()
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    if wav.size == 0:
        raise ValueError(f"empty audio: {path}")
    rms = float(np.sqrt(np.mean(np.square(wav)) + 1e-12))
    peak = float(np.max(np.abs(wav)))
    clipping = float(np.mean(np.abs(wav) >= 0.999))
    non_silent = librosa.effects.split(wav, top_db=45)
    active = sum(int(end - start) for start, end in non_silent)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=wav)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=wav, sr=sr)))
    f0, voiced, probability = librosa.pyin(
        wav,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    finite = np.isfinite(f0)
    voiced_f0 = f0[finite]
    if voiced_f0.size:
        median_f0 = float(np.median(voiced_f0))
        q10, q90 = np.quantile(voiced_f0, [0.10, 0.90])
        span = float(12.0 * np.log2(max(q90, 1e-6) / max(q10, 1e-6)))
    else:
        median_f0 = math.nan
        span = math.nan
    return {
        "duration_seconds": float(wav.size / sr),
        "rms_db": float(20.0 * np.log10(rms + 1e-12)),
        "peak": peak,
        "clipping_ratio": clipping,
        "active_ratio": float(active / wav.size),
        "spectral_flatness": flatness,
        "spectral_centroid_hz": centroid,
        "median_f0_hz": median_f0,
        "f0_span_semitones": span,
        "voiced_ratio": float(np.mean(voiced)) if voiced is not None else 0.0,
        "f0_confidence": float(np.nanmean(probability)) if probability is not None else 0.0,
    }


def quality_score(features: dict[str, float]) -> float:
    clipping = min(1.0, features["clipping_ratio"] / 0.001)
    silence = max(0.0, 0.45 - features["active_ratio"]) / 0.45
    noisy = min(1.0, features["spectral_flatness"] / 0.12)
    uncertain = max(0.0, 0.70 - features["f0_confidence"]) / 0.70
    return max(0.0, min(1.0, 1.0 - 0.35 * clipping - 0.20 * silence - 0.20 * noisy - 0.25 * uncertain))


def _rank_percentiles(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.5] * len(values)
    denominator = max(1, len(values) - 1)
    for rank, index in enumerate(order):
        result[index] = rank / denominator
    return result


def build_prompt_bank(
    source: str | Path,
    output: str | Path,
    *,
    clip_seconds: float = 12.0,
    hop_seconds: float = 12.0,
    min_seconds: float = 8.0,
    ffmpeg: str = "ffmpeg",
) -> Path:
    source = Path(source).resolve()
    output = Path(output).resolve()
    clips = output / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(source)],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(probe.stdout.strip())
    rows: list[dict[str, Any]] = []
    start = 0.0
    index = 1
    while start + min_seconds <= duration:
        actual = min(clip_seconds, duration - start)
        target = clips / f"prompt_{index:03d}_{start:08.2f}s.wav"
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{actual:.3f}", "-ac", "1", "-ar", "44100", "-af", "volume=0.95", str(target)],
            check=True,
        )
        features = analyze_audio(target)
        rows.append({"prompt_id": f"P{index:03d}", "path": str(target), "start_seconds": round(start, 3), **features})
        start += hop_seconds
        index += 1
    if not rows:
        raise ValueError("source is shorter than min_seconds")
    register = _rank_percentiles([float(row["median_f0_hz"]) if math.isfinite(float(row["median_f0_hz"])) else 0.0 for row in rows])
    energy = _rank_percentiles([float(row["rms_db"]) for row in rows])
    for idx, row in enumerate(rows):
        row["register_percentile"] = register[idx]
        row["energy_percentile"] = energy[idx]
        row["quality_score"] = quality_score(row)
    manifest = output / "prompt_index.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest


def f0_retention(reference: str | Path, candidate: str | Path, *, sr: int = 22050) -> dict[str, float]:
    librosa, np = _audio_modules()
    ref, _ = librosa.load(str(reference), sr=sr, mono=True)
    cand, _ = librosa.load(str(candidate), sr=sr, mono=True)
    n_audio = min(ref.size, cand.size)
    ref = ref[:n_audio]
    cand = cand[:n_audio]
    f0_ref, _, _ = librosa.pyin(ref, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    f0_cand, _, _ = librosa.pyin(cand, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    n = min(len(f0_ref), len(f0_cand))
    f0_ref, f0_cand = f0_ref[:n], f0_cand[:n]
    ref_voiced, cand_voiced = np.isfinite(f0_ref), np.isfinite(f0_cand)
    joint = ref_voiced & cand_voiced
    if int(joint.sum()) < 5:
        return {"f0_corr": math.nan, "cent_rmse": math.nan, "uv_mismatch": float(np.mean(ref_voiced != cand_voiced))}
    cents = 1200.0 * np.log2((f0_cand[joint] + 1e-6) / (f0_ref[joint] + 1e-6))
    return {
        "f0_corr": float(np.corrcoef(f0_ref[joint], f0_cand[joint])[0, 1]),
        "cent_rmse": float(np.sqrt(np.mean(np.square(cents)))),
        "uv_mismatch": float(np.mean(ref_voiced != cand_voiced)),
    }


def evaluate_manifest(manifest: str | Path, output: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(manifest).open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        if record.get("status") != "succeeded":
            continue
        candidate = record["output_path"]
        features = analyze_audio(candidate)
        metrics = f0_retention(record["source_path"], candidate)
        rows.append({**record, **features, **metrics, "quality_score": quality_score(features)})
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return rows


def rerank(rows: Iterable[dict[str, Any]], *, identity_scores: dict[str, float] | None = None) -> list[dict[str, Any]]:
    ranked = []
    identity_scores = identity_scores or {}
    for row in rows:
        identity = float(identity_scores.get(str(Path(row["output_path"]).resolve()), identity_scores.get(row["output_path"], 0.0)))
        f0_corr = max(0.0, float(row.get("f0_corr", 0.0))) if math.isfinite(float(row.get("f0_corr", 0.0))) else 0.0
        cent = float(row.get("cent_rmse", 1200.0))
        uv = float(row.get("uv_mismatch", 1.0))
        quality = float(row.get("quality_score", 0.0))
        retention = 0.45 * f0_corr + 0.30 * math.exp(-cent / 250.0) + 0.25 * max(0.0, 1.0 - uv / 0.25)
        score = 0.55 * identity + 0.25 * retention + 0.20 * quality if identity_scores else 0.65 * retention + 0.35 * quality
        ranked.append({**row, "identity_similarity": identity if identity_scores else "", "retention_score": retention, "rerank_score": score})
    return sorted(ranked, key=lambda item: float(item["rerank_score"]), reverse=True)
