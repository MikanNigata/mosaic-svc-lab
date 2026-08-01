# Mosaic-SVC v2 Architecture

## Definition

Mosaic-SVC is a backend-agnostic experiment and retrieval layer for high-quality, offline, zero-shot singing voice conversion.

It does not begin by training a new universal generator or a target-specific model. It keeps the selected SVC backend frozen and improves how target-speaker evidence is enrolled, selected, and evaluated.

```text
source singing
  -> source phrase analysis
  -> Acoustic Memory retrieval
  -> frozen zero-shot SVC backend
  -> optional multi-candidate generation
  -> Identity Memory output reranking
  -> selected output
```

## Objective

Preserve from the source:

- lyrics and pronunciation
- F0, rhythm, and timing
- vibrato and dynamics
- phrase-level expression

Replace from the target:

- timbre and vocal-tract characteristics
- harmonic structure
- register-dependent color
- breathiness, nasality, and resonance

Realtime conversion is explicitly outside the initial scope.

## Quality-partitioned dual memory

### Acoustic Memory

Acoustic Memory contains reference audio that may be supplied directly to a backend. It is built primarily from short, high-quality target singing.

Each entry should retain the original WAV and derived metadata:

- pitch/register position
- F0 span
- energy
- voiced ratio
- phonation/register annotations when reliable
- audio quality
- backend-specific utility observed in downstream generation

### Identity Memory

Identity Memory is built from longer material, including acoustically poor dialogue. It is used for identity verification and ranking, not as an acoustic reconstruction target.

Allowed uses:

- prompt identity scoring
- output identity scoring
- outlier and wrong-speaker detection
- recording-condition clustering
- confidence estimation

Disallowed by default:

- direct reference audio for generation
- mel or waveform reconstruction targets
- vocoder training targets
- unfiltered target fine-tuning

## Progressive enrollment

Mosaic must degrade safely to ordinary zero-shot inference.

```text
5-15 seconds       -> ordinary zero-shot reference
30 seconds-minutes -> Prompt Bank and source-conditioned retrieval
minutes-hours      -> denser retrieval, Identity Memory, output reranking
sufficient HQ data -> optional lightweight adaptation, only after zero-shot evaluation
```

No target-specific adaptation is part of P0-P3.

## Backend boundary

Backends run in separate processes and environments. Mosaic exchanges only job manifests, audio paths, result manifests, and metrics.

```text
Mosaic Core
  -> command manifest
  -> Seed-VC environment
  -> HQ-SVC environment
  -> SoulX-Singer environment
```

The memory schema should remain backend-independent. Prompt utility scores and command adapters are backend-specific.

## Roadmap

### P0: Fixed Seed baseline

- Seed-VC 44.1 kHz
- P05, 48-60 s, 12-second reference
- diffusion steps 60
- CFG 0.50
- F0 condition enabled
- no adapter

### P1-BACKEND: frozen zero-shot comparison

Compare Seed-VC and HQ-SVC with the same source segments and the same P05 reference. Add SoulX-Singer-SVC only when another model family is needed.

### P2-REFERENCE: Acoustic Memory retrieval

Compare:

- R0: fixed P05
- R1: one global-best reference
- R2: source-conditioned top-1 retrieval
- R3: top-3 generation and human oracle selection

The first retriever uses target-relative register, F0 span, energy, and quality. It does not inject frame-level nearest-neighbor features into the generator.

### P3-IDENTITY: dual-memory hypothesis

With the backend still frozen, compare:

- M1: Acoustic Memory retrieval only
- M2: Identity Memory used for reference retrieval
- M3: Identity Memory used only for output reranking
- M4: Identity Memory used for both retrieval and output reranking

The critical comparison is whether long, low-quality identity material improves unseen-song identity without reducing naturalness.

### P4-ADAPT: optional

Only consider lightweight target adaptation when naturalness, pronunciation, pitch, timing, and register behavior are already sufficient and identity remains the isolated weakness.

## Evaluation rules

- Separate development, validation, and final test material.
- Do not evaluate a reference on the same source used to select it.
- Use multiple source singers.
- Record random seeds and backend revisions.
- Loudness-normalize only listening copies; retain untouched raw outputs.
- Blind filenames and keep the mapping private until ratings are complete.
- Treat F0 and UV metrics as failure diagnostics, not identity or naturalness scores.
