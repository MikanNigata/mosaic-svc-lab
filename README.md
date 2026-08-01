# Mosaic-SVC Lab

日本語 | [English](README.en.md)

Mosaic-SVC Lab は、**高品質なオフライン歌唱変換**のための実験・検索・評価リポジトリです。

現在はSeed-VC改造を中心にせず、複数のゼロショットSVCを共通条件で比較し、入力に適したReferenceを検索するバックエンド非依存レイヤーとして再設計しています。

> 固定した汎用ゼロショットSVCへ、品質ごとに役割を分けた話者メモリから最適な参照を供給し、必要に応じて複数出力を評価する。

## 目標

入力歌唱から保持するもの:

- 歌詞・発音
- 音程・リズム・タイミング
- ビブラート、強弱、歌い回し

対象人物から置き換えるもの:

- 声質、声道・共鳴特性
- 倍音構造
- 地声・ミックス・裏声ごとの音色
- 息成分、鼻腔感

リアルタイム性と対象人物ごとの全面fine-tuningは、初期段階の対象外です。

## Mosaicの中心仮説

### Quality-partitioned dual memory

```text
高品質な短い歌唱
  -> Acoustic Memory
  -> 生成器へ渡せるReference Bank

長時間の低品質音声を含む素材
  -> Identity Memory
  -> 本人確認、Prompt選択、出力rerank
```

低品質雑談は、原則として生成用Referenceや音響再構成教師には使いません。長時間データから本人性の統計だけを利用し、ノイズ、部屋、マイク特性の模倣を避けます。

### Progressive enrollment

```text
5-15秒          通常のゼロショット
30秒-数分       Prompt Bankと動的検索
数分-数十分以上 Identity Memoryと複数候補選抜
十分なHQデータ  必要な場合だけ軽量適応
```

### Downstream-feedback retrieval

将来はembedding類似度だけでなく、固定SVCが実際に生成した結果の本人度・自然さ・歌唱保持・artifact・ブラインド選好からRetrieverを改善します。Generator自体はまず固定します。

## RVCとの違い

| | RVC | Mosaic-SVC |
| --- | --- | --- |
| 人物ごとの資産 | 学習済みモデル＋特徴index | 軽量なSpeaker Memory |
| 個人学習 | 原則必要 | P0-P3では行わない |
| 検索 | content特徴を局所的に置換・混合 | 生成用Referenceの選択と出力評価 |
| 低品質長時間音声 | 学習に混ぜると劣化要因 | Identity Memoryに限定 |
| Generator | 対象人物ごと | 全人物で共有するゼロショットbackend |

## 現在の固定baseline

Seed-VC実験でブラインド選好された条件を固定標準器として残します。

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

Adapter版と24秒PromptはF0/UV指標で良くても、聴感ではP05 12秒rawに勝ちませんでした。F0指標は破綻検出には使いますが、本人度・自然さの最終順位には使いません。

## ロードマップ

### P0 — Seed baseline（完了）

現在のP05条件を固定し、Seed側へ追加改造しません。

### P1-BACKEND — 基盤モデル比較

Mosaic補正も個人適応も入れず、同一Source・同一P05で比較します。

```text
Seed-VC P05
vs
HQ-SVC P05
```

必要になった場合だけSoulX-Singer-SVCを追加します。各backendは別環境・別プロセスで動かします。

### P2-REFERENCE — Reference検索

```text
R0 固定P05
R1 development set上のglobal best
R2 入力条件によるtop-1
R3 top-3生成＋人間Oracle
```

最初の検索軸はtarget-relative register、F0幅、energy、qualityです。frame-level特徴をGeneratorへ直接注入しません。

### P3-IDENTITY — Mosaicの中心仮説

固定backendのまま、低品質雑談由来のIdentity MemoryをPrompt検索と出力rerankへ分離して評価します。

```text
M1 Acoustic retrievalのみ
M2 IdentityをPrompt検索に追加
M3 Identityを出力rerankにのみ追加
M4 検索とrerankの両方
```

未知曲でM2/M3/M4がM1より本人度を上げ、自然さを下げなければ中心仮説が成立します。

### P4-ADAPT — 必要時のみ

自然さ、発音、F0、タイミング、声区が十分で、本人性だけが不足する場合に限って、小型adapterやLoRAを検討します。

## 実装されたツール

Python 3.10以上、標準ライブラリのみで動きます。

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

### Backend比較ランナー

```powershell
mosaic-lab run configs/experiments/p1_hq_baseline.example.json --dry-run
mosaic-lab run configs/experiments/p1_hq_baseline.example.json --fail-fast
```

実行コマンド、入力SHA-256、Reference、seed、所要時間、ログ、出力を`manifest.jsonl`へ保存します。

### ブラインド試聴セット

```powershell
mosaic-lab blind `
  --manifest experiments/P1-HQ-001/manifest.jsonl `
  --output experiments/P1-HQ-001/listening `
  --normalize
```

### Promptランキング

```powershell
mosaic-lab rank `
  --source-features configs/retrieval/source_features.example.json `
  --prompt-index configs/retrieval/prompt_index.example.jsonl `
  --weights configs/retrieval/p2_retrieval.example.json `
  --top-k 3
```

詳しい手順は[`docs/experiments/P1_P2_RUNBOOK.md`](docs/experiments/P1_P2_RUNBOOK.md)、設計は[`docs/architecture/MOSAIC_SVC_V2.md`](docs/architecture/MOSAIC_SVC_V2.md)を参照してください。

## リポジトリの役割

```text
mosaic-svc-lab
  実験定義、Memory仕様、Retriever、評価、ブラインド試聴

MikanNigata/seed-vc
  既存Seed baselineと過去のMosaic拡張コード

HQ-SVC / SoulX-Singer
  独立したbackend環境
```

大きな音声、元データ、モデル重み、checkpoint、仮想環境はコミットしません。小さな比較サンプルだけを例外として保存します。

## 旧研究資料

Seed-VC上のPrompt Adapter、CAMPPlus profile、Prompt選択実験は失敗を含む重要な履歴として残します。

- [`docs/architecture/MOSAIC_SVC_R16.md`](docs/architecture/MOSAIC_SVC_R16.md)
- [`docs/experiments/2026-07-31-prompt-selection.md`](docs/experiments/2026-07-31-prompt-selection.md)
- [`configs/current_best.yaml`](configs/current_best.yaml)
