# Mosaic-SVC Lab

English | [日本語](README.ja.md)

Mosaic-SVC Lab is an experimental repository for high-quality singing voice conversion built around a forked Seed-VC runtime.

The practical target is simple:

> Make a source guide vocal sing with the perceived identity of a target singer, while preserving pitch, rhythm, lyrics, and singing expression.

This repository tracks the research design, experiment logs, current best settings, lightweight listening samples, and orchestration scripts. The runnable Seed-VC implementation lives in a separate fork.

```text
runtime fork: https://github.com/MikanNigata/seed-vc
lab repo:     https://github.com/MikanNigata/mosaic-svc-lab
local root:   D:\voice-lab
```

## Motivation

Initial experiments showed that a small adapter on the CAMPPlus/global-style path did not produce a clearly audible improvement. Blind listening preferred a raw Seed-VC prompt over numerically better adapter or longer-prompt variants.

This changed the priority of the project:

```text
before: speaker adapter / memory first
after:  canonical prompt bank first, adapter second
```

The current working hypothesis is:

> In Seed-VC singing conversion, target identity is strongly controlled by the reference prompt path, especially prompt mel and prompt semantic features. Therefore prompt selection must be treated as a first-class speaker asset, not as a convenience input.

## Seed-VC Conditioning Model

The 44.1 kHz Seed-VC singing model conditions target identity through three reference-derived paths:

```text
reference audio
  -> Whisper semantic prompt condition
  -> prompt mel
  -> CAMPPlus global style vector
```

For source singing \(x\), reference prompt \(r\), and target style vector \(s\), the current abstraction is:

```math
\hat{y} = G_{\theta}
\left(
  C(x),
  F_0(x),
  P_{\mathrm{sem}}(r),
  P_{\mathrm{mel}}(r),
  S_{\mathrm{camp}}(r)
\right)
```

where:

- \(C(x)\): Seed-VC source content features.
- \(F_0(x)\): source pitch trajectory after Seed-VC length regulation.
- \(P_{\mathrm{sem}}(r)\): prompt semantic condition from reference audio.
- \(P_{\mathrm{mel}}(r)\): prompt mel frames.
- \(S_{\mathrm{camp}}(r)\): 192-dimensional CAMPPlus global style embedding.
- \(G_{\theta}\): frozen Seed-VC acoustic generator and vocoder.

The project originally tested whether a low-rank style/prompt adapter could move identity without damaging F0:

```math
H_t =
W_{\mathrm{base}}
\left[
  x_t,\,
  p_t,\,
  c_t,\,
  s
\right]
+ \lambda\,B(A(z_r))
```

where \(z_r\) is a prompt summary, \(A\) and \(B\) are low-rank projections, and \(\lambda\) is an inference-time adapter strength.

The first results indicate that prompt selection itself is currently more audible than this adapter.

## Current Best Setting

Current preferred setting from blind listening:

```yaml
condition: P05_12s_raw
prompt_source: dadadada_tenshi_vocal.wav
prompt_start_seconds: 48
prompt_duration_seconds: 12
diffusion_steps: 60
inference_cfg_rate: 0.50
adapter: none
```

Local command:

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

## Experiments

### P0: Frozen Seed-VC Baseline

Purpose:

```text
Measure what can be improved without changing Seed-VC's content encoder, generator, or vocoder.
```

Conditions tested:

| ID | Condition | Purpose |
| --- | --- | --- |
| A | Seed-VC zero-shot | Baseline |
| B | Fixed canonical prompt | Prompt selection effect |
| C | Style adapter | CAMPPlus/style-path adaptation effect |
| D0 | Inference-only prototype correction | CAMPPlus prototype effect |
| M1 | Prompt bank selection | Prompt mel/semantic effect |
| M2 | Prompt Adapter | Prompt-path residual effect |
| M4 | Dialogue speaker profile rerank | Use low-quality speech only as identity signal |

### Prompt Bank Experiment

Prompt candidates were cut from high-quality target singing. The most relevant candidates were:

| Candidate | Segment | Notes |
| --- | ---: | --- |
| P05 | 48s-60s | Blind listening winner |
| P06 | 60s-72s | Previously perceived as slightly clearer |
| P48_72 | 48s-72s | Numerically strong, not preferred in blind listening |

### Dialogue Speaker Profile

The 25-minute dialogue material was not used as acoustic training data.

It was used to build a CAMPPlus speaker centroid:

```math
\bar{s}_{\mathrm{dialogue}}
=
\operatorname{Normalize}
\left(
  \frac{1}{|K|}
  \sum_{i \in K}
  S_{\mathrm{camp}}(d_i)
\right)
```

where \(K\) contains the retained non-outlier dialogue chunks.

The prompt rerank score was:

```math
\operatorname{score}(r)
=
\alpha
\cos
\left(
  S_{\mathrm{camp}}(r),
  \bar{s}_{\mathrm{dialogue}}
\right)
+
\beta Q(r)
```

with \(\alpha=0.70\), \(\beta=0.30\), and \(Q(r)\) as an audio-quality score.

Observed reranking:

| Rank | Prompt | CAMPPlus Similarity | Combined Score |
| ---: | --- | ---: | ---: |
| 1 | prompt_05_048.00s | 0.425142 | 0.594224 |
| 2 | prompt_07_072.00s | 0.423787 | 0.584651 |
| 3 | prompt_08_084.00s | 0.387137 | 0.554121 |
| 4 | prompt_03_024.00s | 0.483816 | 0.545296 |
| 5 | prompt_06_060.00s | 0.346047 | 0.529108 |

### Blind Listening Result

Blind set:

```text
samples/blind_ittai40_p05_tests/
```

Mapping:

| Blind File | Actual Condition |
| --- | --- |
| test_01.mp3 | P48_72 24s raw |
| test_02.mp3 | P05 12s + Prompt Adapter strength 0.5 |
| test_03.mp3 | P05 12s raw |

User preference:

```text
test_03.mp3 -> P05 12s raw
```

### Metrics

Metrics were useful for debugging but did not fully predict listening preference.

| Condition | F0 Corr | Cent RMSE | UV Mismatch | Listening |
| --- | ---: | ---: | ---: | --- |
| P05 12s raw | 0.968344 | 92.98 | 0.139872 | Preferred |
| P05 12s adapter strength 0.5 | 0.996016 | 48.94 | 0.123041 | Not clearly better |
| P05 12s adapter strength 1.0 | 0.994381 | 84.41 | 0.224898 | Too strong |
| P48_72 24s raw | 0.996700 | 44.93 | 0.073418 | Not preferred |

This is an important negative result:

```text
better F0/UV metrics != better perceived identity or naturalness
```

## Current Research Direction

The next useful work is not more random adapter tuning. It is a more systematic prompt-bank study.

Planned next steps:

1. Cut 20-50 high-quality prompt candidates.
2. Generate the same evaluation clip for each prompt.
3. Normalize loudness with LUFS, not peak normalization.
4. Blind-rank prompts by perceived identity and naturalness.
5. Analyze the winning prompts by register, energy, phonation, silence ratio, and CAMPPlus similarity.
6. Only after a stable prompt bank exists, distill or adapt its behavior into an adapter.

## Repository Layout

```text
configs/
  current_best.yaml
docs/
  architecture/
    MOSAIC_SVC_R16.md
  experiments/
    2026-07-31-prompt-selection.md
samples/
  blind_ittai40_p05_tests/
scripts/
  run_current_best.ps1
  build_dialogue_profile.ps1
  rank_prompts_by_dialogue_profile.ps1
```

## Data Policy

This repository intentionally does not track:

- source datasets
- long generated WAV files
- model checkpoints
- pretrained weights
- virtual environments

Small MP3 comparison clips may be tracked when they are useful for experiment review.

## Status

Current status:

```text
Prompt-bank selection is the strongest practical lever found so far.
Prompt Adapter exists, but is secondary until it wins blind listening.
Dialogue data is useful as a speaker-profile signal, not as acoustic training data.
```

