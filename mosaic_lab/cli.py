from __future__ import annotations

import argparse
import json
from pathlib import Path

from .blind import load_successful_results, prepare_blind_set
from .experiment import load_experiment, plan_jobs, run_jobs
from .retrieval import dump_ranking, load_json, load_jsonl, rank_prompts


def _command_run(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    jobs = plan_jobs(config, config_path=args.config)
    manifest = args.manifest or Path(config.get("output_root", "experiments")) / str(config["experiment_id"]) / "manifest.jsonl"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mosaic-lab", description="Mosaic-SVC backend comparison and retrieval utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a backend comparison experiment from JSON")
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
