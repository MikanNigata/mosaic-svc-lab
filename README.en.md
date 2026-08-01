# Mosaic-SVC Lab

[日本語](README.md) | English

Mosaic-SVC Lab is an experiment, retrieval, and evaluation repository for **high-quality offline singing voice conversion**.

It is being redesigned from a Seed-VC-specific extension into a backend-agnostic layer that compares frozen zero-shot SVC systems and selects references suited to each source phrase.

> Supply a frozen universal zero-shot SVC backend with references selected from quality-partitioned speaker memory, then evaluate multiple outputs when necessary.

## Core hypothesis

```text
short high-quality singing
  -> Acoustic Memory
  -> references that may be supplied to the generator

long material including low-quality dialogue
  -> Identity Memory
  -> identity verification, prompt selection, and output reranking
```

Low-quality dialogue is not used as a generation reference or acoustic reconstruction target by default. Mosaic aims to extract identity evidence without copying noise, room response, or microphone coloration.

## Difference from RVC

RVC creates a target-specific model and feature index. Mosaic initially keeps a shared zero-shot backend frozen and creates lightweight target-specific memory. Retrieval selects whole generation references and ranks outputs rather than replacing frame-level content features.

## Fixed Seed baseline

```yaml
backend: Seed-VC 44.1kHz
prompt: P05, 48-60 seconds
prompt_duration: 12 seconds
diffusion_steps: 60
inference_cfg_rate: 0.50
f0_condition: true
adapter: none
```

This condition won the current blind comparison even though adapter and longer-prompt variants produced better F0/UV metrics.

## Roadmap

- **P0:** freeze the existing Seed P05 baseline.
- **P1-BACKEND:** compare Seed-VC and HQ-SVC with the same source and reference, without Mosaic corrections or target adaptation.
- **P2-REFERENCE:** compare fixed P05, a global-best reference, source-conditioned top-1 retrieval, and top-3 human oracle selection.
- **P3-IDENTITY:** test whether long low-quality material improves reference selection or output identity ranking while the backend remains frozen.
- **P4-ADAPT:** consider lightweight adaptation only when identity is the isolated remaining weakness.

## Tools

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v

mosaic-lab run configs/experiments/p1_hq_baseline.example.json --dry-run
mosaic-lab rank `
  --source-features configs/retrieval/source_features.example.json `
  --prompt-index configs/retrieval/prompt_index.example.jsonl `
  --weights configs/retrieval/p2_retrieval.example.json `
  --top-k 3
```

The experiment runner records commands, input hashes, references, random seeds, timing, logs, and canonical output paths in JSONL. The blind-set command randomizes filenames and can apply two-pass ffmpeg loudness normalization to listening copies while preserving raw outputs.

See [`docs/architecture/MOSAIC_SVC_V2.md`](docs/architecture/MOSAIC_SVC_V2.md) and [`docs/experiments/P1_P2_RUNBOOK.md`](docs/experiments/P1_P2_RUNBOOK.md).

Legacy Seed-VC adapter and prompt-selection notes remain in the repository as research history.
