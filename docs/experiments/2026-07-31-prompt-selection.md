# 2026-07-31 Prompt Selection Experiments

## Question

Can target-speaker similarity and naturalness improve without adding much more data by choosing better Seed-VC reference prompts?

## Data

Target high-quality singing:

```text
D:\voice-lab\data\target_clean\dadadada_tenshi_vocal.wav
```

Guide vocal:

```text
D:\voice-lab\data\guide_vocals\ittai_itsukara_head_40s.wav
```

Dialogue profile source:

```text
D:\voice-lab\out\dialogue\maneki_karaoke_stream\maneki_dialogue_strict_denoised.wav
```

## Experiments

### Prompt Sweep

Generated 12-second prompt candidates from the high-quality target singing.

Important candidates:

```text
P05: 48s-60s
P06: 60s-72s
P48_72: 48s-72s, 24-second prompt
```

### Dialogue Speaker Profile

Built a CAMPPlus profile from 25-minute dialogue:

```text
D:\voice-lab\out\mosaic_svc\speaker_profiles\maneki_dialogue25_campplus.pt
```

95 chunks were extracted. 66 were retained for the robust centroid.

Top prompt rerank by dialogue profile:

```text
01 prompt_05_048.00s sim=0.425142 score=0.594224
02 prompt_07_072.00s sim=0.423787 score=0.584651
03 prompt_08_084.00s sim=0.387137 score=0.554121
04 prompt_03_024.00s sim=0.483816 score=0.545296
05 prompt_06_060.00s sim=0.346047 score=0.529108
```

### Prompt Adapter

Trained a small Prompt Adapter for 300 steps on high-quality singing clips only.

The adapter improved some F0 metrics, but listening difference was small.

### Blind Listening

Blind set:

```text
samples/blind_ittai40_p05_tests
```

Mapping:

```text
test_01 = P48_72_24s_raw
test_02 = P05_12s_adapter_s050
test_03 = P05_12s_raw
```

User preferred:

```text
test_03 = P05_12s_raw
```

## Metrics

40-second comparison:

```text
P05 12s raw:
  f0_corr=0.968344
  cent_rmse=92.98
  uv_mismatch=0.139872

P05 12s Prompt Adapter strength 0.5:
  f0_corr=0.996016
  cent_rmse=48.94
  uv_mismatch=0.123041

P48_72 24s raw:
  f0_corr=0.996700
  cent_rmse=44.93
  uv_mismatch=0.073418
```

Listening did not follow the numeric metric ranking. The preferred blind sample was P05 12s raw.

## Conclusion

Prompt choice is currently more important than small adapter changes, but numeric metrics alone are not enough.

Current practical default:

```text
P05 12s raw
cfg=0.50
steps=60
```

Next step:

Build a larger prompt bank and blind-rank it. Treat prompt selection as a first-class part of the architecture.

