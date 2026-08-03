# P1/P2 Seed-Only Runbook

## Status

HQ-SVC and Mosaic-SVC R1.6 P11-P16 are retired. This runbook uses frozen Seed-VC only.

## Goal

1. Confirm P10 against the fixed P05/P07 Seed baselines on held-out songs.
2. Determine whether source-conditioned reference retrieval beats one fixed reference.
3. Improve guide-vocal separation before adding more model adaptation.

## Install

```powershell
cd D:\voice-lab\mosaic-svc-lab
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m pip install -e ".[audio]"
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Verify the Seed environment and preview the experiment:

```powershell
mosaic-lab doctor configs/experiments/p1_p3_windows.example.json
mosaic-lab run configs/experiments/p1_p3_windows.example.json --dry-run
```

## P1 Seed baseline

Use the same held-out source clips and random seed for:

- S0: fixed P05 reference, no adapter.
- S1: fixed P07 reference, no adapter.
- S2: P10 identity-aware adapter plus the accepted K/V LoRA.

Do not add HQ-SVC, R16, another generator, or a streaming model to this comparison.

## P2 retrieval

```powershell
mosaic-lab rank `
  --source-features configs/retrieval/source_features.example.json `
  --prompt-index configs/retrieval/prompt_index.example.jsonl `
  --weights configs/retrieval/p2_retrieval.example.json `
  --top-k 3 `
  --output experiments/P2-RETRIEVAL-001/top3.json
```

Compare fixed P05, global-best reference, source-conditioned top-1, and top-3 human oracle. Treat objective F0/UV metrics as failure diagnostics, then decide with LUFS-matched blind listening.

## P3 Identity reranking

Dialogue may be used for CAMPPlus identity scoring and reranking, never as an acoustic generation reference:

```powershell
mosaic-lab identity-build --input dialogue.wav --output identity.pt --seed-repo D:/voice-lab/seed-vc
mosaic-lab pipeline configs/experiments/p1_p3_windows.example.json `
  --identity-profile identity.pt --seed-repo D:/voice-lab/seed-vc `
  --blind --normalize --fail-fast
```

## Out of scope

- HQ-SVC
- Mosaic-SVC R1.6 P11-P16
- Streaming Student, AP Head, NSF, and Refiner
- target full fine-tuning
- realtime inference
