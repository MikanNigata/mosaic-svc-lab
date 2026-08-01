from __future__ import annotations

import csv
import json
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


def load_successful_results(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            if value.get("status") == "succeeded" and Path(str(value.get("output_path", ""))).is_file():
                rows.append(value)
    return rows


def _probe_loudnorm(input_path: Path, ffmpeg: str, target_lufs: float, true_peak: float, lra: float) -> dict[str, float]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(input_path),
        "-af",
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:print_format=json",
        "-f",
        "null",
        "-",
    ]
    process = subprocess.run(command, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg loudness analysis failed for {input_path}: {process.stderr[-500:]}")
    objects = _JSON_OBJECT.findall(process.stderr)
    if not objects:
        raise RuntimeError(f"ffmpeg did not return loudnorm JSON for {input_path}")
    measured = json.loads(objects[-1])
    keys = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
    if not keys.issubset(measured):
        raise RuntimeError(f"incomplete loudnorm result for {input_path}: {measured}")
    return {key: float(measured[key]) for key in keys}


def normalize_audio(
    input_path: Path,
    output_path: Path,
    *,
    ffmpeg: str = "ffmpeg",
    target_lufs: float = -18.0,
    true_peak: float = -2.0,
    lra: float = 11.0,
    sample_rate: int = 48000,
) -> None:
    measured = _probe_loudnorm(input_path, ffmpeg, target_lufs, true_peak, lra)
    filter_value = (
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:"
        f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true:print_format=summary"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-af",
        filter_value,
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s24le",
        str(output_path),
    ]
    process = subprocess.run(command, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg normalization failed for {input_path}: {process.stderr[-500:]}")


def prepare_blind_set(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    random_seed: int = 20260801,
    normalize: bool = False,
    ffmpeg: str = "ffmpeg",
    condition_ids: set[str] | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    audio_dir = destination / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    selected = [row for row in rows if condition_ids is None or row.get("condition_id") in condition_ids]
    if not selected:
        raise ValueError("no successful experiment outputs matched the selection")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row["source_id"]), []).append(row)

    rng = random.Random(random_seed)
    mapping: dict[str, Any] = {"random_seed": random_seed, "files": {}}
    rating_rows: list[dict[str, str]] = []
    for pair_index, source_id in enumerate(sorted(grouped), start=1):
        conditions = sorted(grouped[source_id], key=lambda row: (str(row["condition_id"]), int(row.get("seed", 0))))
        rng.shuffle(conditions)
        for item_index, row in enumerate(conditions):
            label = chr(ord("A") + item_index)
            filename = f"test_{pair_index:03d}_{label}.wav"
            source = Path(str(row["output_path"]))
            target = audio_dir / filename
            if normalize:
                normalize_audio(source, target, ffmpeg=ffmpeg)
            else:
                shutil.copy2(source, target)
            mapping["files"][filename] = {
                "source_id": source_id,
                "condition_id": row["condition_id"],
                "backend": row["backend"],
                "seed": row["seed"],
                "original_output": str(source),
            }
        rating_rows.append(
            {
                "source_id": source_id,
                "identity": "",
                "naturalness": "",
                "pronunciation": "",
                "pitch_expression": "",
                "register": "",
                "artifacts": "",
                "overall": "",
                "notes": "",
            }
        )

    (destination / "mapping.private.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (destination / "ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rating_rows[0]))
        writer.writeheader()
        writer.writerows(rating_rows)
    return mapping
