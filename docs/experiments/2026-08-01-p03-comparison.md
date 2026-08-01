# 2026-08-01 P03 comparison

## Goal

Implement and compare the P00-P03 Mosaic-SVC prompt/adaptation stages using the same 40-second guide vocal.

The core question is still practical:

```text
Can we improve perceived target-speaker naturalness without breaking F0, pronunciation, or audio quality?
```

## Stages

| Stage | Condition | Purpose |
| --- | --- | --- |
| P00 | `P00_current_best_p05` | Current best baseline: P05 12s raw |
| P01 | `P01_quality_alt_p06` | Prompt-bank alternate from high-quality singing |
| P02 | `P02_prompt_adapter_p05_s050` | P05 with Prompt Adapter strength 0.50 |
| P03 | `P03_dialogue_rank1_p05` | Dialogue-profile rerank rank 1 |
| P03 | `P03_dialogue_rank2_p07` | Dialogue-profile rerank rank 2, first non-P05 candidate |

`P03_dialogue_rank1_p05` duplicates P00 by design because the dialogue profile currently ranks P05 first. The useful P03 test is whether rank2 P07 adds a better audible alternative.

## Runtime

```powershell
D:\voice-lab\mosaic-svc-lab\scripts\run_p03_comparison.ps1
```

Default output:

```text
D:\voice-lab\out\mosaic_svc\p03_compare_ittai40
```

## Fixed inference settings

```yaml
diffusion_steps: 60
inference_cfg_rate: 0.50
prompt_seconds: 12
f0_condition: true
fp16: true
lufs_target: -20
```

## Evaluation policy

The MP3 files are for blind listening. `p03_eval.csv` is only a do-no-harm screen for F0 and voiced/unvoiced stability. Do not choose a winner from numeric F0 metrics alone.

## 2026-08-01 run

Output root:

```text
D:\voice-lab\out\mosaic_svc\p03_compare_ittai40
```

Listening files:

| Condition | MP3 |
| --- | --- |
| P00 current best | `D:\voice-lab\out\mosaic_svc\p03_compare_ittai40\mp3\P00_current_best_p05.mp3` |
| P01 prompt P06 | `D:\voice-lab\out\mosaic_svc\p03_compare_ittai40\mp3\P01_quality_alt_p06.mp3` |
| P02 adapter P05 s0.50 | `D:\voice-lab\out\mosaic_svc\p03_compare_ittai40\mp3\P02_prompt_adapter_p05_s050.mp3` |
| P03 dialogue rank1 P05 | `D:\voice-lab\out\mosaic_svc\p03_compare_ittai40\mp3\P03_dialogue_rank1_p05.mp3` |
| P03 dialogue rank2 P07 | `D:\voice-lab\out\mosaic_svc\p03_compare_ittai40\mp3\P03_dialogue_rank2_p07.mp3` |

Do-no-harm metrics:

| Condition | F0 corr | Cent RMSE | UV mismatch |
| --- | ---: | ---: | ---: |
| P00 current best P05 | 0.9961 | 49.13 | 0.1965 |
| P01 quality alt P06 | 0.8170 | 208.64 | 0.0662 |
| P02 prompt adapter P05 s0.50 | 0.9964 | 46.78 | 0.1361 |
| P03 dialogue rank1 P05 | 0.9959 | 48.65 | 0.1277 |
| P03 dialogue rank2 P07 | 0.9956 | 49.44 | 0.0737 |

Initial interpretation:

- `P01_quality_alt_p06` is risky despite being a clean prompt candidate because F0 correlation drops heavily and cent RMSE rises.
- `P02_prompt_adapter_p05_s050` passes the basic F0 do-no-harm screen and is the most relevant adapter comparison against P00.
- `P03_dialogue_rank2_p07` passes the F0 screen and is the useful non-P05 dialogue-profile comparison.
- `P00_current_best_p05` and `P03_dialogue_rank1_p05` are intentionally near-duplicates; they validate reproducibility more than model improvement.
