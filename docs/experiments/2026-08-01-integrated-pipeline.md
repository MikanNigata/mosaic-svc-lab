# 2026-08-01 Integrated P1-P3 Pipeline

> **Historical experiment record.** The HQ-SVC condition was retired on 2026-08-04 and cannot be executed by the current runner. Only the Seed-VC findings remain relevant to the active plan.

## Purpose

Verify the complete frozen-backend path: Seed-VC and HQ-SVC generation, objective retention checks, dialogue-derived Identity Memory scoring, automatic reranking, and loudness-normalized blind-set generation.

## Conditions

All conditions used the same 15-second source and 12-second target reference family.

| Condition | Backend | Reference / adaptation |
| --- | --- | --- |
| S_P05 | Seed-VC | P05 raw, CFG 0.50, 60 steps |
| S_P07 | Seed-VC | P07 raw, CFG 0.50, 60 steps |
| S_P05_ADAPTER050 | Seed-VC | P05 Prompt Adapter strength 0.50 |
| HQ_P05 | HQ-SVC | P05 raw |

Identity similarity used the robust CAMPPlus centroid previously extracted from approximately 25 minutes of dialogue. Dialogue was not supplied to either generator.

## Result

| Rank | Condition | Identity | F0 corr | Cent RMSE | UV mismatch | Combined |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | S_P05 | 0.3764 | 0.9896 | 63.82 | 0.1409 | 0.5782 |
| 2 | S_P07 | 0.3131 | 0.9491 | 105.86 | 0.1533 | 0.5269 |
| 3 | S_P05_ADAPTER050 | 0.3320 | 0.6824 | 297.56 | 0.1331 | 0.4850 |
| 4 | HQ_P05 | 0.2621 | 0.5892 | 314.07 | 0.1176 | 0.4374 |

P05 remains the practical baseline. The Prompt Adapter altered identity slightly but caused a large F0-retention regression. HQ-SVC remained weak both objectively and in the prior subjective identity assessment.

## Interpretation

This is an engineering smoke test, not evidence that the combined score predicts human preference. Identity scores are meaningful only relative to candidates produced from the same source and must be calibrated with other-speaker negatives. The blind set remains the decision authority.
