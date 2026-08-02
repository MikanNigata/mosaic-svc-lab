# Mosaic-SVC Lab

[日本語](README.md) | English

**Mosaic-SVC Lab** is a research and experiment repository for improving **high-quality offline singing voice conversion** from a small amount of clean singing plus longer, lower-quality material, starting without target-specific training.

> Keep a universal zero-shot SVC backend frozen, retrieve references suited to each source phrase, optionally generate several candidates, and select the output that best preserves naturalness and target identity.

This is not yet a finished voice-conversion application. It is a backend-agnostic experiment, retrieval, blind-listening, and future speaker-memory layer for systems such as Seed-VC, HQ-SVC, and SoulX-Singer-SVC.

---

## TL;DR

```text
source singing
  -> source phrase analysis
  -> reference retrieval from Acoustic Memory
  -> frozen zero-shot SVC backend
  -> one or more converted candidates
  -> Identity Memory / human listening rerank
  -> selected output
```

Mosaic aims to preserve lyrics, pronunciation, pitch, rhythm, timing, vibrato, dynamics, and phrasing while replacing speaker timbre, resonance, harmonic structure, register-dependent color, breathiness, and related target-speaker characteristics.

Real-time conversion, training a large universal model from scratch, and full target-speaker fine-tuning are outside the initial scope.

---

## Why this project exists

A common target-speaker data distribution is asymmetric:

```text
high-quality singing:           tens of seconds to a few minutes
long low-quality speech/video:  tens of minutes to hours
```

Using only the clean material may not cover the speaker's full range. Using noisy long-form audio directly as an acoustic target risks learning noise, room response, codec artifacts, and microphone coloration.

Mosaic therefore separates the roles:

```text
short high-quality singing
  -> Acoustic Memory
  -> references that may be supplied to the generator

long material including low-quality dialogue
  -> Identity Memory
  -> identity verification, prompt retrieval, and output reranking
```

The core hypothesis is:

> Long low-quality material can improve reference selection and output identity ranking without being used as an acoustic training target or generation reference.

---

## Mosaic is not a new generator

Mosaic initially delegates synthesis quality to existing zero-shot SVC systems.

```text
Mosaic Core
  - experiment manifests
  - backend runner
  - Acoustic Memory
  - reference retriever
  - blind-listening preparation
  - future Identity Memory
  - evaluation and result tracking

Backends
  - Seed-VC
  - HQ-SVC
  - SoulX-Singer-SVC
  - future universal SVC systems
```

Each backend runs in its own environment and process. Mosaic does not force incompatible PyTorch, CUDA, codec, and model dependencies into one Python environment.

---

## Difference from RVC and ordinary zero-shot SVC

| | RVC | Ordinary zero-shot SVC | Mosaic-SVC |
| --- | --- | --- | --- |
| Target-specific asset | trained model + feature index | one short reference | Acoustic / Identity Memory |
| Target training | usually required | none | none in P0-P3 |
| Generator | target-specialized | shared | shared and replaceable |
| Retrieval | local content-feature lookup and mixing | usually manual reference | source-conditioned reference retrieval and output reranking |
| Long low-quality data | risky as training data | usually unused | restricted to Identity Memory |
| More target data | retrain/update index | add references manually | progressively enrich memory and retrieval |

The first Mosaic prototype does not inject frame-level nearest-neighbor features into the generator. It retrieves complete reference WAVs, lets the frozen backend generate high-quality audio, and then ranks candidates.

---

## Progressive enrollment

| Available target data | Intended behavior |
| --- | --- |
| 5-15 seconds | ordinary single-reference zero-shot conversion |
| 30 seconds to a few minutes | prompt bank, quality filtering, source-conditioned retrieval |
| minutes to tens of minutes | denser prompt bank, Identity Memory, multi-candidate selection |
| sufficient clean singing | lightweight adaptation only when evidence shows it is needed |

With too little evidence, the system should safely fall back to ordinary zero-shot conversion instead of applying unreliable memory corrections.

---

## Current evidence

The initial Seed-VC study tested:

- 12-second and 24-second references
- a CAMPPlus dialogue profile
- prompt reranking from 25 minutes of dialogue
- a small Prompt Adapter
- several CFG and diffusion settings
- F0, cent RMSE, and UV metrics
- blind listening

The fixed Seed baseline is:

```yaml
backend: Seed-VC 44.1kHz
prompt: P05
prompt_range: 48-60 seconds
prompt_duration: 12 seconds
diffusion_steps: 60
inference_cfg_rate: 0.50
f0_condition: true
adapter: none
```

Blind listening preferred this raw 12-second prompt over adapter and longer-prompt variants that had better F0/UV metrics.

```text
better F0 metrics
  != better identity
  != better naturalness
  != better overall preference
```

This negative result motivated the shift from Seed-specific adapter tuning to backend comparison and systematic reference design.

Legacy research remains available:

- [`docs/architecture/MOSAIC_SVC_R16.md`](docs/architecture/MOSAIC_SVC_R16.md)
- [`docs/experiments/2026-07-31-prompt-selection.md`](docs/experiments/2026-07-31-prompt-selection.md)
- [`configs/current_best.yaml`](configs/current_best.yaml)

---

## Current implementation status

| Feature | Status |
| --- | --- |
| Frozen Seed P05 baseline | complete |
| Manifest-driven backend runner | implemented |
| Command, hash, seed, timing, and log capture | implemented |
| Blind-listening set generation | implemented |
| Optional two-pass ffmpeg loudness normalization | implemented |
| Top-k retrieval from relative register, F0 span, energy, and quality | prototype implemented |
| Seed-VC / HQ-SVC backend presets | implemented and verified on Windows |
| Backend doctor | CUDA, Python, and ffmpeg checks implemented |
| Automatic prompt slicing and feature extraction | implemented |
| CAMPPlus Identity Memory | dialogue centroid and output scoring implemented |
| F0 / UV / quality evaluation and output reranking | implemented |
| End-to-end generation, evaluation, reranking, and blind set | implemented |
| Style / Prompt Adapter | implemented in the Seed fork; excluded from the current baseline |
| ContentVec + Whisper Teacher / De-Timbre | implemented and GPU-smoke-tested in the Seed fork |
| Path-wise leakage probes / external multi-speaker GRL | implemented; real-data pretraining not run |
| Causal Student / Acoustic Converter / AP Head / NSF | first real-data run failed subjective audio quality; research-only |
| File GUI / microphone conversion | implemented; R16 checkpoints are not the production default |

---

## Integrated pipeline

Install the audio extras in the Seed-VC environment, then use one command for generation, objective checks, identity scoring, reranking, and a loudness-normalized blind set.

```powershell
cd D:\voice-lab\mosaic-svc-lab
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m pip install -e ".[audio]"
mosaic-lab doctor configs\experiments\p1_p3_windows.example.json
mosaic-lab pipeline configs\experiments\p1_p3_windows.example.json `
  --identity-profile out\identity.pt `
  --seed-repo D:\voice-lab\seed-vc `
  --blind --normalize --fail-fast
```

`mosaic-lab enroll` builds a 12-second Prompt Bank from clean singing. `mosaic-lab identity-build` converts long dialogue into a robust CAMPPlus centroid without using that dialogue as a generation reference.

---

## Roadmap

### P0 — Seed baseline

Complete. Keep the preferred Seed P05 setting fixed.

### P1-BACKEND — Frozen backend comparison

Compare identical source and reference inputs without Mosaic corrections or target adaptation.

```text
Seed-VC P05
vs
HQ-SVC P05
```

Add SoulX-Singer-SVC only when another independent backend is needed.

### P2-REFERENCE — Reference retrieval

```text
R0 fixed P05
R1 development-set global best
R2 source-conditioned top-1
R3 top-3 generation + human oracle
```

Initial retrieval signals:

- target-relative register
- F0 span
- energy
- quality

### P3-IDENTITY — Core Mosaic hypothesis

Keep the backend frozen and separate the contribution of Identity Memory:

```text
M1 Acoustic retrieval only
M2 Identity added to prompt retrieval
M3 Identity used only for output reranking
M4 Identity used for both retrieval and reranking
```

The hypothesis passes only if unseen songs show improved target identity without reduced naturalness.

### P4-ADAPT — Only if needed

Consider lightweight speaker/timbre adapters or LoRA only when naturalness, pronunciation, pitch, timing, and register are already adequate and identity is the isolated remaining weakness.

---

## Quick start

Mosaic Core requires Python 3.10+ and uses only the standard library.

Additional tools:

- backend-specific environments for actual conversion
- `ffmpeg` for optional listening-copy loudness normalization
- a separate Seed-VC environment
- initially a WSL/Linux environment for HQ-SVC

```powershell
git clone https://github.com/MikanNigata/mosaic-svc-lab.git
cd mosaic-svc-lab

python -m pip install -e .
python -m unittest discover -s tests -v
mosaic-lab --help
```

---

## P1 backend experiment runner

Copy and edit the example manifest:

```powershell
Copy-Item `
  configs/experiments/p1_hq_baseline.example.json `
  configs/experiments/p1_hq_baseline.local.json
```

Inspect planned jobs:

```powershell
mosaic-lab run `
  configs/experiments/p1_hq_baseline.local.json `
  --dry-run
```

Execute:

```powershell
mosaic-lab run `
  configs/experiments/p1_hq_baseline.local.json `
  --fail-fast
```

Each manifest record includes experiment and condition IDs, backend, source, reference, seed, command, SHA-256 hashes, timestamps, elapsed time, return code, logs, and canonical output path.

---

## Blind-listening preparation

```powershell
mosaic-lab blind `
  --manifest experiments/P1-HQ-001/manifest.jsonl `
  --output experiments/P1-HQ-001/listening `
  --normalize
```

Normalization applies only to listening copies; raw backend outputs are preserved. The condition mapping is written separately and should remain hidden during listening.

Recommended axes:

- target identity
- naturalness
- lyrics and pronunciation
- pitch and timing
- vibrato and dynamics
- register preservation
- identity stability across range
- metallic noise, roughness, doubled voice, and other artifacts
- overall preference

---

## P2 prompt ranking

The current retriever is a non-learned prototype:

```powershell
mosaic-lab rank `
  --source-features configs/retrieval/source_features.example.json `
  --prompt-index configs/retrieval/prompt_index.example.jsonl `
  --weights configs/retrieval/p2_retrieval.example.json `
  --top-k 3
```

Current signals and default weights:

```text
0.35 target-relative register
0.25 F0-span match
0.20 energy match
0.20 quality
```

The retriever compares relative positions within the source and target ranges instead of matching absolute F0 directly. The fixed formula is only a baseline for testing whether source-conditioned reference selection has value.

---

## Backend contract

Mosaic does not import backend internals. A thin backend CLI should accept at least:

```text
source path
reference path
output path
random seed
backend-specific settings
```

Conceptually:

```text
mosaic-lab
  -> subprocess / wsl.exe
  -> backend-specific CLI
  -> output.wav
  -> manifest.jsonl
```

For HQ-SVC, the plan is to validate the official-style environment first and then add a thin non-interactive WSL adapter. Native Windows porting and tight coupling to internal APIs are deferred until quality is proven.

---

## Experiment principles

- Separate Development, Validation, and Test sets.
- Do not evaluate final performance on the song used to choose the prompt.
- Use multiple source singers to detect source-timbre leakage.
- Cross prompts with the same random-seed set for stochastic backends.
- Normalize listening level, but do not hide model artifacts with heavy post-processing.
- Use objective pitch/UV metrics for failure detection, not as the final ranking.
- Use blind listening for identity, naturalness, register preservation, and artifacts.

---

## Go / No-Go

Continue when:

- dynamic top-1 is stable against a global-best prompt on unseen songs
- top-3 frequently contains the human-preferred prompt
- Identity Memory improves target identity
- naturalness does not decrease
- small-data behavior does not underperform ordinary zero-shot conversion
- retrieval becomes more stable as target data grows

Redesign when:

- the same prompt always wins
- retrieval scores are unrelated to human preference
- dialogue identity mostly captures recording environment
- additional data does not improve retrieval
- a new backend cannot beat the frozen Seed baseline
- even top-3 human oracle selection cannot beat a fixed prompt

---

## Repository layout

```text
mosaic-svc-lab/
├─ configs/
│  ├─ current_best.yaml
│  ├─ experiments/
│  └─ retrieval/
├─ docs/
│  ├─ architecture/
│  └─ experiments/
├─ mosaic_lab/
│  ├─ cli.py
│  ├─ experiment.py
│  ├─ blind.py
│  └─ retrieval.py
├─ samples/
├─ scripts/
├─ tests/
└─ pyproject.toml
```

Repository roles:

```text
mosaic-svc-lab
  experiment definitions, memory specifications, retrieval, evaluation

MikanNigata/seed-vc
  frozen Seed baseline and legacy Mosaic extensions

HQ-SVC / SoulX-Singer-SVC
  independent backend environments
```

---

## Data policy and responsible use

The repository does not track source datasets, long generated WAV files, private speaker memory, checkpoints, pretrained weights, virtual environments, or private blind mappings. Small curated comparison MP3 files may be committed as exceptions.

Use voice conversion only with appropriate consent and after checking copyright, publicity/personality rights, platform terms, and applicable law. This repository is not intended for impersonation, deception, or rights infringement.

**No repository-wide license has been selected yet.** Public visibility does not itself grant permission to redistribute, commercially use, or create derivatives. Backend code and model weights also have their own licenses.

---

## Documentation

- [Mosaic-SVC v2 Architecture](docs/architecture/MOSAIC_SVC_V2.md)
- [P1 / P2 Runbook](docs/experiments/P1_P2_RUNBOOK.md)
- [Legacy R1.6 Architecture](docs/architecture/MOSAIC_SVC_R16.md)
- [Seed Prompt Selection Experiments](docs/experiments/2026-07-31-prompt-selection.md)
- [P4-P8 Frozen Seed-VC Adaptation](docs/experiments/2026-08-02-p4-p8-adaptation.md)

---

## Immediate next tasks

1. Evaluate the selected P8 condition on held-out full songs for long-form, high-register, and ending artifacts.
2. Integrate F0 correlation, cent RMSE, and UV error into the generation pipeline.
3. Run LUFS-matched blind listening between P8 and the fixed Seed baseline.
4. Calibrate the high-quality singing identity profile with other-speaker negatives.
5. Proceed to identity-aware loss and the Streaming Student only if P8 also wins subjective evaluation.

The current practical candidate is P8:

> Can independently trained K/V-only LoRA and global Style-Slice adapters improve identity and quality together while the Seed-VC base remains frozen?
