# P4-P8 Frozen Seed-VC Adaptation

## Objective

Test whether small target-speaker modules can improve identity without changing the frozen Seed-VC content, F0, mel head, or vocoder paths.

All comparisons use the same source, canonical prompt, diffusion seed, CFG 0.50, and 60 steps. Generalization uses three unseen 15-second clips from different songs.

## Critical implementation fix

The original adapter trainers called `model.cfm.eval()`. A legacy positional argument then reached `DiT.forward` as a tensor-valued `mask_content`, which caused the prompt/content/style condition to be masked during training. Training now uses `model.cfm.train()`, and inference masking only accepts an explicit boolean.

Random state is reset immediately before diffusion inference, so loading an optional adapter cannot change the initial diffusion noise.

## Experiments

| Stage | Trainable path | Result |
|---|---|---|
| P4 | Prompt-mel LoRA | No-Go: deterministic difference was about -55 dB |
| P5 | Layer 8 K/V LoRA, rank 4 | No-Go: deterministic difference was about -61 dB |
| P6 | Layers 4/8/12 K/V LoRA, rank 8 | Singing identity improved; quality decreased |
| P7 | Global CAMPPlus style-slice adapter, rank 4 | Quality improved; singing identity stayed approximately neutral |
| P8 | P6 step 600 + P7 step 600 | Best balance; selected default |

## Evaluation profile correction

The 25-minute dialogue profile is noisy and register-mismatched. It remains useful as an auxiliary speech-identity measure, but it must not be the primary singing metric.

The primary profile is a robust CAMPPlus centroid built from 13 high-quality singing segments, retaining the best 9 segments after outlier removal.

## Three-clip means

| Condition | Singing identity cosine | Quality score |
|---|---:|---:|
| Frozen baseline | 0.712374 | 0.920236 |
| P6 step 600 | 0.721376 | 0.917249 |
| P6 step 800 | 0.726470 | 0.914777 |
| P7 step 600 | 0.712688 | 0.929731 |
| P7 step 800 | 0.713065 | 0.930829 |
| P8: P6-600 + P7-600 | **0.724428** | **0.925862** |
| P8: P6-800 + P7-600 | 0.726403 | 0.923043 |

P8 step 600/600 is selected because it improves both metrics over baseline and keeps more quality margin than the stronger K/V checkpoint.

## Current decision

Use the fixed P07 high-register canonical prompt, K/V LoRA layers 4/8/12 step 600, and Style-Slice Adapter step 600. Keep all base Seed-VC weights frozen. The adaptation assets are small standalone `.pt` files and should be evaluated on new songs before any full-length render.

P8 is a Conditional Go, not proof of final perceptual quality. Required next checks are blind listening, explicit F0/UV scoring, consonant/ending preservation, and full-song artifact review.
