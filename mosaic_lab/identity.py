from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def _run_seed_module(seed_repo: Path, module: str, arguments: list[str], python: Path | None = None) -> None:
    executable = python or seed_repo / ".venv" / "Scripts" / "python.exe"
    process = subprocess.run(
        [str(executable), "-m", module, *arguments],
        cwd=seed_repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip() or f"{module} failed")


def build_identity_profile(input_audio: Path, output: Path, seed_repo: Path, python: Path | None = None) -> None:
    _run_seed_module(seed_repo, "mosaic_svc.p0.build_speaker_profile", ["--input", str(input_audio), "--output", str(output)], python)


def score_manifest(manifest: Path, profile: Path, output: Path, seed_repo: Path, python: Path | None = None) -> dict[str, float]:
    _run_seed_module(
        seed_repo,
        "mosaic_svc.p0.score_outputs_by_profile",
        ["--manifest", str(manifest), "--profile", str(profile), "--output", str(output)],
        python,
    )
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(Path(row["output_path"]).resolve()): float(row["identity_similarity"]) for row in rows}


def write_candidate_manifest(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.get("status") == "succeeded":
                handle.write(json.dumps({"output_path": record["output_path"]}, ensure_ascii=False) + "\n")
