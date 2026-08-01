from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def build_backend_command(
    backend_id: str,
    definition: Mapping[str, Any],
    condition: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    config_dir: Path,
) -> tuple[tuple[str, ...], Path, str | None]:
    kind = str(definition.get("kind", backend_id)).lower()
    repo = _resolve(config_dir, str(definition["repo"]))
    settings = dict(definition.get("defaults", {}))
    settings.update(condition.get("settings", {}))

    if kind == "seed-vc":
        python = _resolve(repo, str(definition.get("python", ".venv/Scripts/python.exe")))
        command = [
            str(python),
            "-m",
            str(definition.get("module", "mosaic_svc.p0.infer_p0")),
            "--source",
            str(context["source"]),
            "--prompt",
            str(context["reference"]),
            "--output",
            str(context["output_dir"]),
            "--diffusion-steps",
            str(settings.get("diffusion_steps", 60)),
            "--inference-cfg-rate",
            str(settings.get("cfg_rate", 0.50)),
            "--prompt-seconds",
            str(settings.get("prompt_seconds", 12.0)),
            "--f0-condition",
            str(settings.get("f0_condition", True)),
            "--fp16",
            str(settings.get("fp16", True)),
        ]
        optional = {
            "style_audio": "--style-audio",
            "style_adapter": "--style-adapter",
            "prompt_adapter": "--prompt-adapter",
            "prompt_adapter_strength": "--prompt-adapter-strength",
            "prototype_bank": "--prototype-bank",
            "prototype_strength": "--prototype-strength",
        }
        for key, flag in optional.items():
            if settings.get(key) is not None:
                command.extend([flag, str(settings[key])])
        return tuple(command), repo, str(context["output_dir"]) + "/**/*.wav"

    if kind == "hq-svc":
        script = _resolve(repo, str(definition.get("script", "run_windows_infer.ps1")))
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Source",
            str(context["source"]),
            "-Target",
            str(context["reference"]),
            "-Output",
            str(context["output_file"]),
        ]
        if settings.get("auto_f0", False):
            command.append("-AutoF0")
        if settings.get("skip_download", True):
            command.append("-SkipDownload")
        if "shift_key" in settings:
            command.extend(["-ShiftKey", str(settings["shift_key"])])
        return tuple(command), repo, None

    raise ValueError(f"unsupported backend kind {kind!r} for {backend_id!r}")


def doctor(backends: Mapping[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for backend_id, raw in backends.items():
        definition = dict(raw)
        repo = Path(str(definition.get("repo", ""))).resolve()
        kind = str(definition.get("kind", backend_id)).lower()
        checks: list[tuple[str, bool, str]] = [("repo", repo.is_dir(), str(repo))]
        if kind == "seed-vc":
            python = _resolve(repo, str(definition.get("python", ".venv/Scripts/python.exe")))
            checks.append(("python", python.is_file(), str(python)))
            if python.is_file():
                process = subprocess.run(
                    [str(python), "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available())"],
                    cwd=repo,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                checks.append(("cuda", process.returncode == 0 and "True" in process.stdout, process.stdout.strip() or process.stderr.strip()))
        elif kind == "hq-svc":
            script = _resolve(repo, str(definition.get("script", "run_windows_infer.ps1")))
            python = _resolve(repo, str(definition.get("python", ".venv/Scripts/python.exe")))
            checks.extend([("script", script.is_file(), str(script)), ("python", python.is_file(), str(python))])
        results.append({"backend": backend_id, "kind": kind, "ok": all(item[1] for item in checks), "checks": checks})
    results.append({"backend": "system", "kind": "tools", "ok": shutil.which("ffmpeg") is not None, "checks": [("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not found")]})
    return results
