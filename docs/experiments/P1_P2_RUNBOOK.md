# P1/P2 Runbook

## Goal

Answer two questions before personal adaptation:

1. Does HQ-SVC provide a better frozen zero-shot baseline than the current Seed-VC P05 condition?
2. Does source-conditioned reference retrieval outperform a fixed or global-best reference?

## Install the lab tools

```powershell
cd D:\voice-lab\mosaic-svc-lab
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m pip install -e .
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Core orchestration uses the Python standard library. Install `.[audio]` for Prompt Bank feature extraction and objective audio metrics. Backend model dependencies remain isolated.

## P1-HQ

Copy and edit:

```text
configs/experiments/p1_hq_baseline.example.json
```

The current Windows port provides non-interactive Seed-VC and HQ-SVC backend presets. Use `p1_p3_windows.example.json` as the working example and verify both environments first:

```powershell
mosaic-lab doctor configs/experiments/p1_p3_windows.example.json
```

Preview commands without executing them:

```powershell
mosaic-lab run configs/experiments/p1_hq_baseline.example.json --dry-run
```

Run and preserve a JSONL manifest:

```powershell
mosaic-lab run configs/experiments/p1_hq_baseline.example.json --fail-fast
```

Expected comparison:

```text
S0_seed_p05 vs H0_hqsvc_p05
x E1 mid, E2 high/mix, E3 falsetto, E4 fast consonants
```

## Blind listening set

```powershell
mosaic-lab blind `
  --manifest experiments/P1-HQ-001/manifest.jsonl `
  --output experiments/P1-HQ-001/listening `
  --conditions S0_seed_p05,H0_hqsvc_p05 `
  --normalize
```

`--normalize` applies two-pass ffmpeg loudnorm to listening copies only. Raw backend outputs remain unchanged.

The command creates:

```text
listening/audio/test_001_A.wav
listening/audio/test_001_B.wav
listening/mapping.private.json
listening/ratings.csv
```

Do not inspect `mapping.private.json` before completing ratings.

## P2 retrieval prototype

Start with manually or externally extracted features. The prototype intentionally separates feature extraction from ranking so that RMVPE, librosa, or another analyzer can be substituted later without changing the experiment format.

Example:

```powershell
mosaic-lab rank `
  --source-features configs/retrieval/source_features.example.json `
  --prompt-index configs/retrieval/prompt_index.example.jsonl `
  --weights configs/retrieval/p2_retrieval.example.json `
  --top-k 3 `
  --output experiments/P2-RETRIEVAL-001/E2_top3.json
```

Initial score:

```text
0.35 register match
0.25 F0-span match
0.20 energy match
0.20 quality
```

Use target-relative register percentiles rather than absolute source F0.

## P2 conditions

- R0: fixed P05
- R1: global best on development material
- R2: top-1 from the retriever
- R3: generate top-3 and select the human oracle

Interpretation:

- R2 > R1: dynamic retrieval is promising.
- R3 > R1 but R2 <= R1: the bank has value; ranking is weak.
- R1 dominates: use one stable reference for this backend.
- R3 cannot beat R0: stop before Identity Memory and revisit the backend or bank.

## P3 Identity reranking

Build Identity Memory from dialogue without using it as a generation reference:

```powershell
mosaic-lab identity-build --input dialogue.wav --output identity.pt --seed-repo D:/voice-lab/seed-vc
```

Run generation, objective evaluation, identity scoring, reranking, and blind-set creation:

```powershell
mosaic-lab pipeline configs/experiments/p1_p3_windows.example.json `
  --identity-profile identity.pt `
  --seed-repo D:/voice-lab/seed-vc `
  --blind --normalize --fail-fast
```

Identity scoring is a candidate-selection signal, not an acceptance oracle. Validate against blind human preference and include other-speaker negatives before tuning its weight.

## Still out of scope

- target full fine-tuning
- phoneme-aware frame retrieval
- realtime inference
- GUI work
