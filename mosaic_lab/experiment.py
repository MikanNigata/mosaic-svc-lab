from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .backends import build_backend_command


@dataclass(frozen=True)
class PlannedJob:
    experiment_id: str
    condition_id: str
    backend: str
    source_id: str
    source_path: Path
    reference_id: str
    reference_path: Path
    seed: int
    command: tuple[str, ...]
    cwd: Path | None
    environment: dict[str, str]
    output_dir: Path
    output_file: Path
    collect_glob: str | None


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format(value: str, context: Mapping[str, Any], field: str) -> str:
    try:
        return value.format_map(context)
    except KeyError as exc:
        raise ValueError(f"unknown placeholder {exc.args[0]!r} in {field}") from exc


def load_experiment(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a JSON object")
    return config


def plan_jobs(config: Mapping[str, Any], *, config_path: str | Path | None = None) -> list[PlannedJob]:
    experiment_id = _require_nonempty_string(config.get("experiment_id"), "experiment_id")
    raw_output_root = Path(_require_nonempty_string(config.get("output_root", "experiments"), "output_root"))
    config_dir = Path(config_path).resolve().parent if config_path else Path.cwd()
    output_root = raw_output_root if raw_output_root.is_absolute() else (config_dir / raw_output_root).resolve()

    def resolve_path(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (config_dir / candidate).resolve()

    sources = _require_mapping(config.get("sources"), "sources")
    references = _require_mapping(config.get("references"), "references")
    backends = config.get("backends", {})
    if not isinstance(backends, Mapping):
        raise ValueError("backends must be an object")
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("conditions must be a non-empty array")

    seeds = config.get("seeds", [1234])
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be a non-empty integer array")

    jobs: list[PlannedJob] = []
    for raw_condition in conditions:
        condition = _require_mapping(raw_condition, "condition")
        if condition.get("enabled", True) is False:
            continue
        condition_id = _require_nonempty_string(condition.get("id"), "condition.id")
        backend = _require_nonempty_string(condition.get("backend"), f"{condition_id}.backend")
        reference_id = _require_nonempty_string(condition.get("reference"), f"{condition_id}.reference")
        if reference_id not in references:
            raise ValueError(f"condition {condition_id} references unknown reference {reference_id!r}")
        reference_path = resolve_path(_require_nonempty_string(references[reference_id], f"references.{reference_id}"))
        raw_command = condition.get("command")
        if raw_command is not None and (not isinstance(raw_command, list) or not raw_command or not all(isinstance(part, str) for part in raw_command)):
            raise ValueError(f"condition {condition_id}.command must be a non-empty string array")
        if raw_command is None and backend not in backends:
            raise ValueError(f"condition {condition_id} has no command and references unknown backend {backend!r}")

        condition_env = condition.get("env", {})
        if not isinstance(condition_env, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in condition_env.items()):
            raise ValueError(f"condition {condition_id}.env must be a string map")

        for source_id, source_value in sources.items():
            source_id = _require_nonempty_string(source_id, "source id")
            source_path = resolve_path(_require_nonempty_string(source_value, f"sources.{source_id}"))
            for seed in seeds:
                job_root = (output_root / experiment_id / "raw" / source_id / condition_id / f"seed_{seed}").resolve()
                output_file = job_root / "output.wav"
                context = {
                    "experiment_id": experiment_id,
                    "condition_id": condition_id,
                    "backend": backend,
                    "source_id": source_id,
                    "source": str(source_path),
                    "reference_id": reference_id,
                    "reference": str(reference_path),
                    "seed": seed,
                    "output_dir": str(job_root),
                    "output_file": str(output_file),
                    "config_dir": str(config_dir),
                }
                if raw_command is None:
                    command, cwd, default_collect = build_backend_command(
                        backend,
                        _require_mapping(backends[backend], f"backends.{backend}"),
                        condition,
                        context,
                        config_dir=config_dir,
                    )
                else:
                    command = tuple(_format(part, context, f"{condition_id}.command") for part in raw_command)
                    cwd_value = condition.get("cwd")
                    cwd = resolve_path(_format(cwd_value, context, f"{condition_id}.cwd")) if isinstance(cwd_value, str) else None
                    default_collect = None
                environment = {key: _format(value, context, f"{condition_id}.env.{key}") for key, value in condition_env.items()}
                collect_value = condition.get("collect_glob")
                collect_glob = _format(collect_value, context, f"{condition_id}.collect_glob") if isinstance(collect_value, str) else default_collect
                jobs.append(
                    PlannedJob(
                        experiment_id=experiment_id,
                        condition_id=condition_id,
                        backend=backend,
                        source_id=source_id,
                        source_path=source_path,
                        reference_id=reference_id,
                        reference_path=reference_path,
                        seed=seed,
                        command=command,
                        cwd=cwd,
                        environment=environment,
                        output_dir=job_root,
                        output_file=output_file,
                        collect_glob=collect_glob,
                    )
                )
    if not jobs:
        raise ValueError("experiment contains no enabled jobs")
    return jobs


def _collect_output(job: PlannedJob) -> Path:
    if job.output_file.is_file():
        return job.output_file
    if not job.collect_glob:
        raise FileNotFoundError(f"backend did not create {job.output_file} and collect_glob is unset")
    candidates = [Path(item) for item in glob.glob(job.collect_glob, recursive=True) if Path(item).is_file()]
    if not candidates:
        raise FileNotFoundError(f"collect_glob matched no files: {job.collect_glob}")
    source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    job.output_file.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != job.output_file.resolve():
        shutil.copy2(source, job.output_file)
    return job.output_file


def run_jobs(
    jobs: Iterable[PlannedJob],
    *,
    manifest_path: str | Path,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with manifest.open("a", encoding="utf-8") as handle:
        for job in jobs:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            started_at = _utc_now()
            started = time.perf_counter()
            stdout_path = job.output_dir / "stdout.log"
            stderr_path = job.output_dir / "stderr.log"
            return_code: int | None = None
            status = "planned" if dry_run else "running"
            error: str | None = None

            if not dry_run:
                environment = os.environ.copy()
                environment.update(job.environment)
                try:
                    process = subprocess.run(
                        list(job.command),
                        cwd=str(job.cwd) if job.cwd else None,
                        env=environment,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        capture_output=True,
                        check=False,
                    )
                    return_code = process.returncode
                    stdout_path.write_text(process.stdout, encoding="utf-8")
                    stderr_path.write_text(process.stderr, encoding="utf-8")
                    if process.returncode != 0:
                        raise RuntimeError(f"backend exited with code {process.returncode}")
                    _collect_output(job)
                    status = "succeeded"
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"

            elapsed = time.perf_counter() - started
            result = {
                "experiment_id": job.experiment_id,
                "condition_id": job.condition_id,
                "backend": job.backend,
                "source_id": job.source_id,
                "source_path": str(job.source_path),
                "source_sha256": _sha256(job.source_path),
                "reference_id": job.reference_id,
                "reference_path": str(job.reference_path),
                "reference_sha256": _sha256(job.reference_path),
                "seed": job.seed,
                "command": list(job.command),
                "cwd": str(job.cwd) if job.cwd else None,
                "output_path": str(job.output_file),
                "output_sha256": _sha256(job.output_file),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "started_at": started_at,
                "elapsed_seconds": round(elapsed, 6),
                "return_code": return_code,
                "status": status,
                "error": error,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            results.append(result)
            if status == "failed" and fail_fast:
                break
    return results
