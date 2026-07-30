# Mosaic-SVC Lab

Mosaic-SVC Lab is the experiment and design repository for a Seed-VC-based singing voice conversion workflow.

The current goal is practical:

- Keep F0, rhythm, and pronunciation from the source singing.
- Improve target-speaker similarity using high-quality references.
- Avoid training on low-quality dialogue as an acoustic target.
- Keep experiments reproducible on Windows 11 + RTX 3070 8GB.

## Repository Role

This repository is not a fork of Seed-VC.

Seed-VC remains the runnable upstream/forked dependency at:

```text
D:\voice-lab\seed-vc
https://github.com/MikanNigata/seed-vc
```

This repository tracks:

- Mosaic-SVC architecture notes.
- Experiment plans and results.
- Small orchestration scripts.
- Lightweight comparison samples.
- Current best settings.

Large generated audio, model checkpoints, pretrained weights, virtual environments, and source datasets stay local.

## Current Practical Finding

The most useful discovery so far is that Seed-VC output quality is strongly affected by the reference prompt path:

```text
reference audio -> prompt semantic + prompt mel + CAMPPlus style
```

In blind listening, the current preferred result was:

```text
P05 12s raw
cfg=0.50
steps=60
no adapter
```

This means prompt selection currently matters more than the first small Prompt Adapter.

## Current Best Local Command

```powershell
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m mosaic_svc.p0.infer_p0 `
  --source D:\voice-lab\data\guide_vocals\ittai_itsukara_head_40s.wav `
  --prompt D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_05_048.00s.wav `
  --output D:\voice-lab\out\mosaic_svc\p0\current_best_p05_raw `
  --diffusion-steps 60 `
  --inference-cfg-rate 0.50 `
  --prompt-seconds 12 `
  --f0-condition True `
  --fp16 True
```

## Key Files

- `docs/architecture/MOSAIC_SVC_R16.md`
- `docs/experiments/2026-07-31-prompt-selection.md`
- `configs/current_best.yaml`
- `scripts/`
- `samples/blind_ittai40_p05_tests/`

