from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from .temporal_memory import iter_memory_records, load_memory_metadata
from .temporal_retrieval import load_query_records, summarize_query_records


def _output_paths(output: str | Path) -> tuple[Path | None, Path | None, Path]:
    destination = Path(output).resolve()
    suffix = destination.suffix.lower()
    if suffix == ".html":
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination, None, destination.with_suffix(".summary.json")
    if suffix == ".png":
        destination.parent.mkdir(parents=True, exist_ok=True)
        return None, destination, destination.with_suffix(".summary.json")
    if suffix:
        raise ValueError("temporal-visualize output must be a directory, .html, or .png path")
    destination.mkdir(parents=True, exist_ok=True)
    return destination / "temporal_report.html", None, destination / "summary.json"


def _selected_candidate(record: dict[str, Any]) -> dict[str, Any] | None:
    selected = record.get("selected_patch_id")
    if not selected:
        return None
    return next(
        (candidate for candidate in record.get("candidates", []) if candidate.get("patch_id") == selected),
        None,
    )


def _normalize(values: list[float]) -> list[float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [0.5] * len(values)
    low, high = min(finite), max(finite)
    if high - low < 1e-9:
        return [0.5] * len(values)
    return [max(0.0, min(1.0, (value - low) / (high - low))) if math.isfinite(value) else 0.0 for value in values]


def _polyline(values: list[float], *, width: int, height: int, color: str) -> str:
    if not values:
        return ""
    normalized = _normalize(values)
    denominator = max(1, len(values) - 1)
    points = " ".join(
        f"{index * width / denominator:.2f},{height - value * height:.2f}"
        for index, value in enumerate(normalized)
    )
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke" />'


def _build_html(
    records: list[dict[str, Any]],
    memory_records: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    width, height = 1100, 300
    f0 = [float(record.get("source_features", {}).get("f0_median_hz", 0.0)) for record in records]
    register = [float(record.get("source_features", {}).get("relative_register", 0.5)) for record in records]
    energy = [float(record.get("source_features", {}).get("energy_percentile", 0.5)) for record in records]
    confidence = [float(record.get("retrieval_confidence", 0.0)) for record in records]
    target_register: list[float] = []
    distances: list[float] = []
    for record in records:
        candidate = _selected_candidate(record)
        target_register.append(
            float(candidate.get("target_features", {}).get("relative_register", 0.5)) if candidate else 0.5
        )
        distances.append(float(candidate.get("feature_distance", 0.0)) if candidate else 0.0)
    switch_lines: list[str] = []
    previous: str | None = None
    denominator = max(1, len(records) - 1)
    for index, record in enumerate(records):
        selected = record.get("selected_patch_id")
        if previous is not None and selected and selected != previous:
            x = index * width / denominator
            switch_lines.append(f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{height}" stroke="#e24a33" stroke-opacity="0.35" />')
        if selected:
            previous = selected
    chart = "".join(
        [
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Temporal retrieval timeline">',
            '<rect width="100%" height="100%" fill="#fbfaf5" />',
            *[f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="#ddd7ca" />' for y in (0, 75, 150, 225, 300)],
            *switch_lines,
            _polyline(f0, width=width, height=height, color="#005f73"),
            _polyline(register, width=width, height=height, color="#0a9396"),
            _polyline(target_register, width=width, height=height, color="#ee9b00"),
            _polyline(energy, width=width, height=height, color="#bb3e03"),
            _polyline(confidence, width=width, height=height, color="#6a4c93"),
            "</svg>",
        ]
    )
    rows = []
    for record in records[:500]:
        candidate = _selected_candidate(record)
        patch_id = str(record.get("selected_patch_id") or "-")
        memory_record = memory_records.get(patch_id, {})
        rows.append(
            "<tr>"
            f"<td>{int(record.get('frame_index', 0))}</td>"
            f"<td>{float(record.get('source_time_seconds', 0.0)):.2f}</td>"
            f"<td>{html.escape(patch_id)}</td>"
            f"<td>{float(record.get('source_features', {}).get('relative_register', 0.5)):.3f}</td>"
            f"<td>{float(candidate.get('target_features', {}).get('relative_register', 0.5)) if candidate else 0.0:.3f}</td>"
            f"<td>{float(candidate.get('feature_distance', 0.0)) if candidate else 0.0:.3f}</td>"
            f"<td>{float(record.get('retrieval_confidence', 0.0)):.3f}</td>"
            f"<td>{html.escape(str(memory_record.get('audio_path', '')))}</td>"
            "</tr>"
        )
    cards = "".join(
        f'<article><strong>{html.escape(label)}</strong><span>{value}</span></article>'
        for label, value in (
            ("Query frames", summary["query_frames"]),
            ("Selected", summary["frames_with_selection"]),
            ("Mean confidence", f'{summary["mean_confidence"]:.3f}'),
            ("Switches / sec", f'{summary["switches_per_second"]:.3f}'),
            ("Unique patches", summary["unique_patches_used"]),
            ("Mean distance", f'{summary["mean_feature_distance"]:.3f}'),
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mosaic Temporal Timbre Memory P0</title>
<style>
:root{{--paper:#f4f0e6;--ink:#192523;--muted:#62706d;--line:#d8d0c1;--accent:#005f73}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 Georgia,"Yu Mincho",serif}}
main{{max-width:1240px;margin:auto;padding:40px 24px 72px}}h1{{font-size:clamp(30px,5vw,58px);line-height:1;margin:0 0 8px}}
.subtitle{{color:var(--muted);margin-bottom:28px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:20px 0}}
article{{background:#fffdf8;border:1px solid var(--line);padding:15px}}article strong,article span{{display:block}}article span{{font:700 25px/1.2 ui-monospace,monospace;margin-top:8px;color:var(--accent)}}
.chart{{background:#fffdf8;border:1px solid var(--line);padding:14px;overflow:auto}}svg{{display:block;min-width:800px;width:100%;height:auto}}
.legend{{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0 30px;font-family:ui-monospace,monospace;font-size:12px}}
.legend i{{display:inline-block;width:20px;height:3px;margin-right:6px;vertical-align:middle}}
table{{width:100%;border-collapse:collapse;background:#fffdf8;font-family:ui-monospace,monospace;font-size:12px}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left}}th{{position:sticky;top:0;background:#e9e3d6}}
.table-wrap{{max-height:620px;overflow:auto;border:1px solid var(--line)}}code{{font-family:ui-monospace,monospace}}@media(max-width:640px){{main{{padding:24px 12px}}}}
</style></head><body><main>
<h1>Temporal Timbre Memory <em>P0</em></h1>
<p class="subtitle">Source-relative retrieval path · memory patches {int(metadata.get('accepted_patch_count', 0))} accepted / {int(metadata.get('patch_count', 0))} total</p>
<section class="cards">{cards}</section>
<section class="chart">{chart}</section>
<div class="legend"><span><i style="background:#005f73"></i>Source F0</span><span><i style="background:#0a9396"></i>Source register</span><span><i style="background:#ee9b00"></i>Target register</span><span><i style="background:#bb3e03"></i>Energy</span><span><i style="background:#6a4c93"></i>Confidence</span><span><i style="background:#e24a33"></i>Patch switch</span></div>
<h2>Retrieval path</h2><div class="table-wrap"><table><thead><tr><th>Frame</th><th>Time</th><th>Patch</th><th>Src reg</th><th>Tgt reg</th><th>Distance</th><th>Confidence</th><th>Audio</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
</main></body></html>"""


def _write_png(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "PNG temporal visualization requires matplotlib. Install with:\n\n"
            'python -m pip install -e ".[visualization]"\n\n'
            "HTML reports do not require matplotlib."
        ) from exc
    times = [float(record.get("source_time_seconds", 0.0)) for record in records]
    register = [float(record.get("source_features", {}).get("relative_register", 0.5)) for record in records]
    energy = [float(record.get("source_features", {}).get("energy_percentile", 0.5)) for record in records]
    confidence = [float(record.get("retrieval_confidence", 0.0)) for record in records]
    figure, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(times, register, color="#0a9396", label="source register")
    axes[1].plot(times, energy, color="#bb3e03", label="source energy")
    axes[2].plot(times, confidence, color="#6a4c93", label="retrieval confidence")
    for axis in axes:
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Source time (seconds)")
    figure.suptitle(
        f"Mosaic TTM-P0 · {summary['patch_switches']} switches · "
        f"mean confidence {summary['mean_confidence']:.3f}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def visualize_temporal_query(
    query: str | Path,
    memory: str | Path,
    output: str | Path,
) -> dict[str, str]:
    """Create an HTML or PNG retrieval report plus a machine-readable summary."""
    records = load_query_records(query)
    metadata = load_memory_metadata(memory)
    memory_records = {str(record["patch_id"]): record for record in iter_memory_records(memory)}
    summary = summarize_query_records(records)
    html_path, png_path, summary_path = _output_paths(output)
    if html_path is not None:
        html_path.write_text(
            _build_html(records, memory_records, summary, metadata),
            encoding="utf-8",
        )
    if png_path is not None:
        _write_png(png_path, records, summary)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "html": str(html_path) if html_path else "",
        "png": str(png_path) if png_path else "",
        "summary": str(summary_path),
    }
