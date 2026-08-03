# Mosaic-SVC R1.6 廃止済みNo-Go記録

> **廃止済み。** 2026-08-04にR16 P11-P16、Streaming、AP、NSF、Refinerを現行計画から完全除外した。全CLIは実行を拒否する。この文書は失敗の再現記録であり、ロードマップではない。

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
- L1 Prototype Memory: P14/P15/P16で共通のbounded style補正として接続済み
- Live Quality Refiner: ゼロ初期化・補正上限付きの独立checkpointとして実装済み
- Leakage評価: linear/MLP probeに加え、verification EER、nearest centroid、ContentVec retentionを実装済み

## 実装と学習済みモデルの区別

P11-P16のコード、checkpoint契約、学習順、ランタイムは実装済みであり、合成データによるCUDA直列スモークで最終WAV生成まで確認済みである。ただし合成スモークcheckpointは音質評価に使えない。

Prototype/Refinerは明示的に指定した場合のみ有効になる。最初のR16実データcheckpointが主観音質No-Goだった事実は変わらず、これらの追加実装だけでP10を置換しない。

追加実装・再学習・再評価は行わない。現行開発はSeed-VC P0-P10、Reference選択、入力分離品質、ブラインド評価に限定する。

## 廃止範囲

P11-P16、Streaming Student、Acoustic Converter、AP Head、NSF、R16 Prototype/Refinerは全て廃止する。P0-P10のSeed-VC適応実験は別系統として維持する。

## Source Of Truth

履歴コードは [`MikanNigata/seed-vc`](https://github.com/MikanNigata/seed-vc) の `mosaic_svc/p11` から `mosaic_svc/p16` に残すが、共通retirement guardにより実行不可とする。
