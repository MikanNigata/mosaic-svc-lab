from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .audio import build_prompt_bank, evaluate_manifest, rerank
from .backends import doctor
from .blind import load_successful_results, prepare_blind_set
from .experiment import load_experiment, plan_jobs, run_jobs
from .identity import build_identity_profile, score_manifest, write_candidate_manifest
from .retrieval import dump_ranking, load_json, load_jsonl, rank_prompts
from .temporal_memory import EnrollmentConfig, build_temporal_memory, load_memory_metadata
from .temporal_retrieval import RetrievalConfig, query_temporal_memory
from .temporal_visualize import visualize_temporal_query


def _default_manifest(config: dict, config_path: Path) -> Path:
    output_root = Path(str(config.get("output_root", "experiments")))
    if not output_root.is_absolute():
        output_root = (config_path.resolve().parent / output_root).resolve()
    return output_root / str(config["experiment_id"]) / "manifest.jsonl"


def _command_run(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    jobs = plan_jobs(config, config_path=args.config)
    if args.manifest:
        manifest = args.manifest
    else:
        manifest = _default_manifest(config, args.config)
    results = run_jobs(jobs, manifest_path=manifest, dry_run=args.dry_run, fail_fast=args.fail_fast)
    summary = {
        "planned": len(results),
        "succeeded": sum(item["status"] == "succeeded" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "dry_run": args.dry_run,
        "manifest": str(manifest),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed"] else 0


def _command_rank(args: argparse.Namespace) -> int:
    source = load_json(args.source_features)
    prompts = load_jsonl(args.prompt_index)
    weights = load_json(args.weights) if args.weights else None
    ranking = rank_prompts(source, prompts, weights=weights, top_k=args.top_k)
    if args.output:
        dump_ranking(args.output, ranking)
    print(
        json.dumps(
            [
                {
                    "rank": index,
                    "prompt_id": item.prompt_id,
                    "score": round(item.score, 8),
                    "components": {key: round(value, 8) for key, value in item.components.items()},
                    "path": item.record.get("path"),
                }
                for index, item in enumerate(ranking, start=1)
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _command_blind(args: argparse.Namespace) -> int:
    rows = load_successful_results(args.manifest)
    conditions = set(args.conditions.split(",")) if args.conditions else None
    mapping = prepare_blind_set(
        rows,
        args.output,
        random_seed=args.seed,
        normalize=args.normalize,
        ffmpeg=args.ffmpeg,
        condition_ids=conditions,
    )
    print(json.dumps({"output": str(args.output), "files": len(mapping["files"])}, ensure_ascii=False, indent=2))
    return 0


def _command_doctor(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    results = doctor(config.get("backends", {}))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item["ok"] for item in results) else 1


def _command_enroll(args: argparse.Namespace) -> int:
    manifest = build_prompt_bank(
        args.source,
        args.output,
        clip_seconds=args.clip_seconds,
        hop_seconds=args.hop_seconds,
        min_seconds=args.min_seconds,
        ffmpeg=args.ffmpeg,
    )
    print(json.dumps({"prompt_index": str(manifest)}, ensure_ascii=False, indent=2))
    return 0


def _command_identity_build(args: argparse.Namespace) -> int:
    build_identity_profile(args.input, args.output, args.seed_repo, args.python)
    print(json.dumps({"identity_profile": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


def _command_temporal_enroll(args: argparse.Namespace) -> int:
    memory_path = build_temporal_memory(
        args.source,
        args.output,
        config=EnrollmentConfig(
            patch_seconds=args.patch_seconds,
            hop_seconds=args.hop_seconds,
            analysis_sr=args.analysis_sr,
            min_active_ratio=args.min_active_ratio,
            min_f0_confidence=args.min_f0_confidence,
            max_clipping_ratio=args.max_clipping_ratio,
        ),
        overwrite=args.overwrite,
    )
    metadata = load_memory_metadata(memory_path)
    print(
        json.dumps(
            {
                "memory": str(memory_path),
                "patches": metadata["patch_count"],
                "accepted": metadata["accepted_patch_count"],
                "rejected": metadata["rejected_patch_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _command_temporal_query(args: argparse.Namespace) -> int:
    query_path = query_temporal_memory(
        args.source,
        args.memory,
        args.output,
        config=RetrievalConfig(
            top_k=args.top_k,
            temperature=args.temperature,
            continuity_weight=args.continuity_weight,
            jump_penalty=args.jump_penalty,
            min_confidence=args.min_confidence,
        ),
        update_seconds=args.update_seconds,
        disable_smoothing=args.disable_smoothing,
    )
    summary_path = query_path.with_suffix(".summary.json")
    print(
        json.dumps(
            {"query": str(query_path), "summary": str(summary_path)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _command_temporal_visualize(args: argparse.Namespace) -> int:
    outputs = visualize_temporal_query(args.query, args.memory, args.output)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


def _write_ranked(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = output.with_suffix(".md")
    lines = ["# Mosaic-SVC ranking", "", "| Rank | Source | Condition | Backend | Score | Identity | F0 corr | Cent RMSE | UV mismatch |", "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for index, row in enumerate(rows, start=1):
        identity = row.get("identity_similarity", "")
        lines.append(
            f"| {index} | {row['source_id']} | {row['condition_id']} | {row['backend']} | "
            f"{float(row['rerank_score']):.4f} | {identity if identity == '' else f'{float(identity):.4f}'} | "
            f"{float(row['f0_corr']):.4f} | {float(row['cent_rmse']):.2f} | {float(row['uv_mismatch']):.4f} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate(args: argparse.Namespace) -> tuple[list[dict], Path]:
    output = args.output or args.manifest.with_name("evaluation.csv")
    rows = evaluate_manifest(args.manifest, output)
    identity_scores = None
    if args.identity_profile:
        candidate_manifest = output.with_name("identity_candidates.jsonl")
        write_candidate_manifest(rows, candidate_manifest)
        identity_output = output.with_name("identity_scores.csv")
        identity_scores = score_manifest(candidate_manifest, args.identity_profile, identity_output, args.seed_repo, args.python)
    ranked = rerank(rows, identity_scores=identity_scores)
    ranking = output.with_name("ranking.csv")
    _write_ranked(ranked, ranking)
    return ranked, ranking


def _command_evaluate(args: argparse.Namespace) -> int:
    ranked, ranking = _evaluate(args)
    print(json.dumps({"evaluated": len(ranked), "ranking": str(ranking)}, ensure_ascii=False, indent=2))
    return 0


def _command_pipeline(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    manifest = _default_manifest(config, args.config)
    results = run_jobs(plan_jobs(config, config_path=args.config), manifest_path=manifest, fail_fast=args.fail_fast)
    if any(item["status"] == "failed" for item in results):
        print(json.dumps({"status": "failed", "manifest": str(manifest)}, ensure_ascii=False, indent=2))
        return 1
    evaluation_args = argparse.Namespace(
        manifest=manifest,
        output=manifest.with_name("evaluation.csv"),
        identity_profile=args.identity_profile,
        seed_repo=args.seed_repo,
        python=args.python,
    )
    ranked, ranking = _evaluate(evaluation_args)
    blind_output = None
    if args.blind:
        blind_output = manifest.with_name("listening")
        prepare_blind_set(load_successful_results(manifest), blind_output, random_seed=args.seed, normalize=args.normalize, ffmpeg=args.ffmpeg)
    print(json.dumps({"status": "succeeded", "jobs": len(results), "ranked": len(ranked), "manifest": str(manifest), "ranking": str(ranking), "listening": str(blind_output) if blind_output else None}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mosaic-lab", description="Mosaic-SVC Seed experiment and retrieval utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a Seed-VC experiment from JSON")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--manifest", type=Path)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.set_defaults(func=_command_run)

    rank_parser = subparsers.add_parser("rank", help="rank Prompt Bank entries for one source feature record")
    rank_parser.add_argument("--source-features", required=True, type=Path)
    rank_parser.add_argument("--prompt-index", required=True, type=Path)
    rank_parser.add_argument("--weights", type=Path)
    rank_parser.add_argument("--top-k", type=int, default=3)
    rank_parser.add_argument("--output", type=Path)
    rank_parser.set_defaults(func=_command_rank)

    blind_parser = subparsers.add_parser("blind", help="build a randomized listening set from successful runs")
    blind_parser.add_argument("--manifest", required=True, type=Path)
    blind_parser.add_argument("--output", required=True, type=Path)
    blind_parser.add_argument("--conditions", help="comma-separated condition IDs")
    blind_parser.add_argument("--seed", type=int, default=20260801)
    blind_parser.add_argument("--normalize", action="store_true", help="apply two-pass ffmpeg loudnorm")
    blind_parser.add_argument("--ffmpeg", default="ffmpeg")
    blind_parser.set_defaults(func=_command_blind)

    doctor_parser = subparsers.add_parser("doctor", help="check configured backend environments")
    doctor_parser.add_argument("config", type=Path)
    doctor_parser.set_defaults(func=_command_doctor)

    enroll_parser = subparsers.add_parser("enroll", help="slice high-quality singing and build a Prompt Bank index")
    enroll_parser.add_argument("--source", required=True, type=Path)
    enroll_parser.add_argument("--output", required=True, type=Path)
    enroll_parser.add_argument("--clip-seconds", type=float, default=12.0)
    enroll_parser.add_argument("--hop-seconds", type=float, default=12.0)
    enroll_parser.add_argument("--min-seconds", type=float, default=8.0)
    enroll_parser.add_argument("--ffmpeg", default="ffmpeg")
    enroll_parser.set_defaults(func=_command_enroll)

    identity_parser = subparsers.add_parser("identity-build", help="build CAMPPlus Identity Memory from long dialogue")
    identity_parser.add_argument("--input", required=True, type=Path)
    identity_parser.add_argument("--output", required=True, type=Path)
    identity_parser.add_argument("--seed-repo", required=True, type=Path)
    identity_parser.add_argument("--python", type=Path)
    identity_parser.set_defaults(func=_command_identity_build)

    temporal_enroll_parser = subparsers.add_parser(
        "temporal-enroll",
        help="build a Temporal Timbre Memory from high-quality target singing",
    )
    temporal_enroll_parser.add_argument("--source", required=True, type=Path)
    temporal_enroll_parser.add_argument("--output", required=True, type=Path)
    temporal_enroll_parser.add_argument("--patch-seconds", type=float, default=0.40)
    temporal_enroll_parser.add_argument("--hop-seconds", type=float, default=0.10)
    temporal_enroll_parser.add_argument("--analysis-sr", type=int, default=22050)
    temporal_enroll_parser.add_argument("--min-active-ratio", type=float, default=0.50)
    temporal_enroll_parser.add_argument("--min-f0-confidence", type=float, default=0.50)
    temporal_enroll_parser.add_argument("--max-clipping-ratio", type=float, default=0.001)
    temporal_enroll_parser.add_argument("--overwrite", action="store_true")
    temporal_enroll_parser.set_defaults(func=_command_temporal_enroll)

    temporal_query_parser = subparsers.add_parser(
        "temporal-query",
        help="retrieve target timbre patches for each source singing frame",
    )
    temporal_query_parser.add_argument("--source", required=True, type=Path)
    temporal_query_parser.add_argument("--memory", required=True, type=Path)
    temporal_query_parser.add_argument("--output", required=True, type=Path)
    temporal_query_parser.add_argument("--top-k", type=int, default=5)
    temporal_query_parser.add_argument("--update-seconds", type=float, default=0.10)
    temporal_query_parser.add_argument("--temperature", type=float, default=0.15)
    temporal_query_parser.add_argument("--continuity-weight", type=float, default=0.25)
    temporal_query_parser.add_argument("--jump-penalty", type=float, default=0.05)
    temporal_query_parser.add_argument("--min-confidence", type=float, default=0.0)
    temporal_query_parser.add_argument("--disable-smoothing", action="store_true")
    temporal_query_parser.set_defaults(func=_command_temporal_query)

    temporal_visualize_parser = subparsers.add_parser(
        "temporal-visualize",
        help="visualize a temporal retrieval path as HTML or PNG",
    )
    temporal_visualize_parser.add_argument("--query", required=True, type=Path)
    temporal_visualize_parser.add_argument("--memory", required=True, type=Path)
    temporal_visualize_parser.add_argument("--output", required=True, type=Path)
    temporal_visualize_parser.set_defaults(func=_command_temporal_visualize)

    evaluate_parser = subparsers.add_parser("evaluate", help="measure F0/UV/quality and rerank successful outputs")
    evaluate_parser.add_argument("--manifest", required=True, type=Path)
    evaluate_parser.add_argument("--output", type=Path)
    evaluate_parser.add_argument("--identity-profile", type=Path)
    evaluate_parser.add_argument("--seed-repo", type=Path, default=Path("D:/voice-lab/seed-vc"))
    evaluate_parser.add_argument("--python", type=Path)
    evaluate_parser.set_defaults(func=_command_evaluate)

    pipeline_parser = subparsers.add_parser("pipeline", help="run generation, evaluation, reranking, and optional blind-set creation")
    pipeline_parser.add_argument("config", type=Path)
    pipeline_parser.add_argument("--identity-profile", type=Path)
    pipeline_parser.add_argument("--seed-repo", type=Path, default=Path("D:/voice-lab/seed-vc"))
    pipeline_parser.add_argument("--python", type=Path)
    pipeline_parser.add_argument("--fail-fast", action="store_true")
    pipeline_parser.add_argument("--blind", action="store_true")
    pipeline_parser.add_argument("--normalize", action="store_true")
    pipeline_parser.add_argument("--seed", type=int, default=20260801)
    pipeline_parser.add_argument("--ffmpeg", default="ffmpeg")
    pipeline_parser.set_defaults(func=_command_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
