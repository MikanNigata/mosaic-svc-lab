# Mosaic-SVC R1.6 実装スナップショット

## 目的

同じ対象話者資産から、Seed-VCによる高品質Renderと、専用Studentによる低遅延Live変換を構築する。

```text
高品質歌唱
  -> Canonical Prompt / CAMPPlus Identity / Prototype
  -> Seed-VC Teacher                    -> Render
  -> Content Teacher -> Causal Student -> Live
```

ソースから音素、F0、リズム、energyを保持し、対象話者から音色と安全な発声傾向だけを与える。R1ではraw mel residual、spectral residual、target mel patch retrievalを使用しない。

## 実装済み経路

### P0-P10: Frozen Seed-VC適応

- Canonical Prompt BankとCAMPPlus profileによる参照選択
- Style-Slice Adapter
- Prompt Adapter
- K/V-only LoRA
- Identity-aware adaptation
- LUFS統一、F0/UV/quality評価、ブラインド比較

現時点の実用baselineはSeed-VC + P05/P07系の選択済み参照である。小型Adapterの改善はNo-Harm範囲だが、参照選択以上の大差はまだ確認できていない。

### P11: Offline Content Teacher

```text
ContentVec fine features --+
                           +-> bounded gated fusion -> De-Timbre Adapter
Whisper semantic anchor ---+
```

- Whisper gateは最大0.30
- De-Timbreはzero-initされたbounded residual
- timbre perturbation consistency、content retention、delta retention
- 外部複数話者だけで使用するwarmup付きGRL
- 対象話者1人の適応ではGRLを開始できない

### P12: Leakage評価

ContentVec、Whisper、Fusion、De-Timbreの各地点へlinear/MLP speaker probeを当てる。最低2話者、各2クリップ未満のmanifestは拒否する。Leakage低下だけでなくcontent保持を同時評価する。

### P13-P16: Streaming経路

```text
80-bin causal input
  -> Causal Content Student
  -> Causal Acoustic Converter + explicit Prosody Bus
  -> Target AP Head
  -> Harmonic-noise NSF
  -> waveform
```

- Student: frame/delta/delta2/long-vowel/boundary蒸留、Dynamic Chunk Training
- Prosody: absolute F0、UV、confidence、slope、energy、phonation
- Converter: 学習済みStudent出力を入力として学習可能
- AP: 8帯域aperiodicity、harmonicity、noise ratio
- NSF: harmonic/noise excitation、F0位相をチャンク間で保持
- Runtime: Live Fast 80 ms、Live Quality 160 ms、Render
- GUI: Gradioファイル変換
- Live: sounddeviceの入出力callbackとGPU workerをqueueで分離

## 実装と学習済みモデルの区別

P11-P16のコード、checkpoint契約、学習順、ランタイムは実装済みであり、合成データによるCUDA直列スモークで最終WAV生成まで確認済みである。ただし合成スモークcheckpointは音質評価に使えない。

実用化には次が残る。

1. 高品質歌唱を楽曲・セッション単位でtrain/validation/testへ確定する。
2. P11を実データで学習し、P12でLeakage/Retentionを判定する。
3. P13、P14、P15 AP、P15 NSFを順番に学習する。
4. 未学習曲でSeed RenderとStreaming経路を比較する。
5. RTF、実測遅延、チャンク境界、本人度、F0、発音、ノイズのGo/No-Goを行う。

## 条件付き機能

Level 2 Mid-block K/V補正はP6に実装済みだが、Level 1/LoRAで局所的な本人度不足が残る場合だけ使用する。Mel/spectral residualはR1.6では実装しない。

## Source Of Truth

実装コードは [`MikanNigata/seed-vc`](https://github.com/MikanNigata/seed-vc) の `mosaic_svc/p11` から `mosaic_svc/p16` に置く。本リポジトリは設計、実験manifest、評価結果、採否判断を管理する。
