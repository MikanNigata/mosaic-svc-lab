# Mosaic-SVC R1.6 Architecture Snapshot

## Objective

Use one target-speaker asset set for both:

- Offline high-quality rendering.
- Future low-latency streaming conversion.

R1.6 separates high-quality render and streaming paths:

```text
common condition extraction + speaker assets
  -> Seed-VC Teacher for render
  -> Streaming Student for live
```

## Core Principle

Preserve from source:

- phoneme/content
- F0
- rhythm
- energy trajectory

Replace from target:

- speaker timbre
- stable style characteristics
- register/phonation tendencies when safe

Do not inject raw mel or spectral residuals in R1.

## P0/P1 Scope

The current implementation deliberately starts smaller than full R1.6:

```text
Frozen Seed-VC
  + Canonical Prompt Bank
  + optional Prompt Adapter
  + optional CAMPPlus profile-based prompt reranking
```

P0/P1 does not implement:

- ContentVec + Whisper teacher fusion.
- De-Timbre Adapter.
- Streaming Student.
- NSF vocoder.
- AP Head.
- Level 2 K/V memory.
- Mel or spectral residual injection.

## Updated Design Finding

Seed-VC 44.1 kHz singing conversion derives target identity from three prompt paths:

```text
reference audio
  -> Whisper semantic condition
  -> prompt mel
  -> CAMPPlus global style embedding
```

The first experiments showed that CAMPPlus-only style correction and small adapters changed output less than expected. Prompt mel/semantic selection was more audible.

Therefore `Canonical Prompt Bank` is promoted to a first-class speaker asset.

## Current Speaker Asset Hierarchy

1. High-quality Prompt Bank
2. Manual/blind prompt rankings
3. CAMPPlus dialogue profile for prompt reranking only
4. Prompt Adapter as secondary experiment
5. Future LoRA/Student/Vocoder work only after prompt bank wins reliably

## Dialogue Data Policy

The 25-minute dialogue material is low acoustic quality.

Use it for:

- CAMPPlus speaker profile
- prompt reranking signal
- negative examples for quality gates
- future robustness checks

Do not use it for:

- mel reconstruction target
- vocoder teacher
- acoustic fine-tuning
- prototype acoustic memory

