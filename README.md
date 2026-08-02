# Mosaic-SVC Lab

日本語 | [English](README.en.md)

**Mosaic-SVC Lab** は、少量の高品質歌唱と、長時間の低品質音声を役割分離して利用し、**個人学習なしを出発点に高品質なオフライン歌唱変換を改善する**ための研究・実験リポジトリです。

> 汎用ゼロショットSVCは固定したまま、入力フレーズに合うReferenceを選び、必要なら複数出力を比較し、対象人物らしさと自然さを高める。

現在は完成した音声変換アプリではありません。Seed-VC、HQ-SVC、SoulX-Singer-SVCなどを交換可能なbackendとして扱い、比較実験、Reference検索、ブラインド試聴、将来のSpeaker Memoryを管理する基盤を作っています。

---

## TL;DR

```text
入力歌唱
  ├─ 歌詞・発音
  ├─ F0・リズム・タイミング
  └─ ビブラート・強弱・歌い回し
            │
            ▼
    入力フレーズを分析
            │
            ▼
Acoustic MemoryからReference検索
            │
            ▼
  固定ゼロショットSVC backend
            │
            ├─ 1候補を生成
            └─ Top-k候補を複数生成
                         │
                         ▼
       Identity Memory / 人間試聴でrerank
                         │
                         ▼
                    変換歌唱
```

Mosaicが置き換えるのは主に次です。

- 話者の声質
- 声道・共鳴特性
- 倍音構造
- 地声、ミックス、裏声ごとの音色
- 息成分、鼻腔感、発声傾向

Mosaicが保持したいのは次です。

- 歌詞と発音
- 音程、リズム、タイミング
- ビブラート、強弱、歌い回し
- 元歌唱の表現とフレージング

初期段階では、リアルタイム化、巨大モデルの独自学習、対象人物ごとの全面fine-tuningは行いません。

---

## なぜ作るのか

一般的な状況では、対象人物について次のようなデータの偏りがあります。

```text
高品質な歌唱       数十秒〜数分
低品質な雑談・映像 数十分〜数時間
```

高品質音声だけでは本人の発声範囲を十分に網羅できません。一方、低品質音声をそのまま音響学習や生成Referenceに使うと、ノイズ、残響、圧縮、マイク特性まで模倣する危険があります。

Mosaicでは両者を同じ用途へ混ぜません。

```text
高品質な短時間歌唱
  -> Acoustic Memory
  -> 実際に生成器へ渡せるReference Bank

低品質を含む長時間音声
  -> Identity Memory
  -> 本人確認、Prompt選択、生成結果のrerank
```

中心仮説は次です。

> 低品質な長時間音声を音響教師にせず、本人性の証拠としてだけ使えば、固定ゼロショットSVCの自然さを壊さずにReference選択と出力選抜を改善できるのではないか。

---

## Mosaicは新しいGeneratorではない

現段階のMosaicは、独自の巨大な歌声生成モデルではありません。

```text
Mosaic Core
  ├─ 実験定義
  ├─ Backend Runner
  ├─ Acoustic Memory
  ├─ Reference Retriever
  ├─ Blind Listening
  ├─ 将来のIdentity Memory
  └─ 評価・結果管理

Backends
  ├─ Seed-VC
  ├─ HQ-SVC
  ├─ SoulX-Singer-SVC
  └─ 将来の汎用SVC
```

生成品質そのものは、まず既存の高性能ゼロショットSVCに任せます。Mosaicは「どのモデルへ、どのReferenceを、どの条件で渡し、どの出力を採用するか」を担当します。

backendは別Python環境・別プロセスで動かします。PyTorch、CUDA、codec、依存ライブラリを一つの環境へ無理に混ぜません。

---

## RVC、通常のゼロショットSVCとの違い

| 項目 | RVC | 通常のゼロショットSVC | Mosaic-SVC |
| --- | --- | --- | --- |
| 対象人物ごとの資産 | 学習済みモデル＋特徴index | 数秒〜数十秒のReference | Acoustic / Identity Memory |
| 個人学習 | 原則必要 | 不要 | P0〜P3では不要 |
| Generator | 人物ごとに専用化 | 全人物で共有 | 全人物で共有、交換可能 |
| Retrieval | content特徴を局所的に検索・混合 | 通常は手動Reference | 入力に応じたReference検索と出力rerank |
| 長時間低品質音声 | 学習へ混ぜると劣化要因 | 通常は利用しない | Identity Memoryに限定 |
| データ増加時 | 再学習・index更新 | 手動Reference追加 | Memoryと検索候補が段階的に充実 |
| 主目的 | 対象人物専用モデル | 手軽なゼロショット変換 | 品質非対称データを学習なしで活用 |

Mosaicの検索は、最初からRVCのようにframe-level特徴をGeneratorへ直接差し込みません。

```text
入力フレーズを分析
  -> 適切なReference WAVを検索
  -> 固定backendで生成
  -> 必要なら複数出力をrerank
```

これにより、検索フレーム間の不連続、リンギング、音色の震えを避け、高品質生成はbackend側へ任せます。

---

## Progressive Enrollment

特定の最低データ量を必須にしません。

| 利用可能データ | 動作 |
| --- | --- |
| 5〜15秒 | 通常の単一Referenceゼロショット |
| 30秒〜数分 | Prompt Bank、品質選別、入力依存検索 |
| 数分〜数十分 | 高密度Prompt Bank、Identity Memory、複数候補選抜 |
| 十分な高品質歌唱 | 必要性が確認された場合だけ軽量適応 |

データが少ない場合は、複雑な処理で悪化させず、通常のゼロショットへ安全に退化する設計を目指します。

---

## 現在までに分かったこと

初期のSeed-VC実験では、次を比較しました。

- 12秒Promptと24秒Prompt
- CAMPPlus話者プロファイル
- 25分の雑談によるPrompt rerank
- 小型Prompt Adapter
- 複数のCFG、diffusion設定
- F0、cent RMSE、UV mismatch
- ブラインド試聴

現在の固定baselineは次です。

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

この条件は、F0/UV指標で優れていたAdapter版や24秒Prompt版より、ブラインド試聴で選ばれました。

```text
F0指標が良い
  != 本人らしい
  != 自然
  != 総合的に好まれる
```

この負の結果から、Seed固有のAdapter調整を続けるより、基盤モデル比較とReference設計を先に行う方針へ変更しました。

旧実験は失敗を含む研究履歴として保存しています。

- [`docs/architecture/MOSAIC_SVC_R16.md`](docs/architecture/MOSAIC_SVC_R16.md)
- [`docs/experiments/2026-07-31-prompt-selection.md`](docs/experiments/2026-07-31-prompt-selection.md)
- [`configs/current_best.yaml`](configs/current_best.yaml)

---

## 現在の実装状況

| 機能 | 状態 |
| --- | --- |
| Seed P05固定baseline | 完了 |
| Manifest駆動のBackend Runner | 実装済み |
| 実行コマンド・hash・seed・ログ記録 | 実装済み |
| ブラインドセット生成 | 実装済み |
| ffmpeg 2-pass LUFS正規化 | 実装済み、任意 |
| 相対声区・F0幅・energy・qualityによるTop-k検索 | Prototype実装済み |
| Seed-VC / HQ-SVC定型backend | Windows実推論まで実装済み |
| backend環境診断 | CUDA / Python / ffmpeg確認を実装済み |
| 自動Prompt切り出し・特徴抽出 | 実装済み |
| CAMPPlus Identity Memory | 雑談centroid構築・出力採点を実装済み |
| F0 / UV / 音質評価と自動rerank | 実装済み |
| 生成・評価・rerank・ブラインド試聴の一括実行 | 実装済み |
| Style / Prompt Adapter | Seed fork側で実装済み、現行baselineでは不採用 |
| ContentVec + Whisper Teacher / De-Timbre | Seed fork側で実装・GPUスモーク済み |
| 経路別Leakage Probe / 外部複数話者GRL | Seed fork側で実装済み、実データ事前学習は未実施 |
| Causal Student / Acoustic Converter / AP Head / NSF | Seed fork側で実装・直列スモーク済み |
| ファイル変換GUI / マイク変換 | Seed fork側で実装済み、実用checkpoint学習前 |

---

## 統合パイプライン

音声指標を使うため、Seed-VCのPython環境へaudio extras込みでインストールします。

```powershell
cd D:\voice-lab\mosaic-svc-lab
D:\voice-lab\seed-vc\.venv\Scripts\python.exe -m pip install -e ".[audio]"
```

backend診断:

```powershell
mosaic-lab doctor configs\experiments\p1_p3_windows.example.json
```

高品質歌唱から12秒Prompt Bankを作成:

```powershell
mosaic-lab enroll --source target_vocal.wav --output out\prompt_bank
```

低品質雑談は生成Referenceにせず、Identity Memoryだけへ登録:

```powershell
mosaic-lab identity-build `
  --input dialogue_25min.wav `
  --output out\identity.pt `
  --seed-repo D:\voice-lab\seed-vc
```

生成、F0/UV・音質評価、Identity採点、rerank、LUFS統一ブラインドセット作成を一括実行:

```powershell
mosaic-lab pipeline configs\experiments\p1_p3_windows.example.json `
  --identity-profile out\identity.pt `
  --seed-repo D:\voice-lab\seed-vc `
  --blind --normalize --fail-fast
```

バックエンドは別venv・別プロセスのままです。実験定義で`command`を手書きする旧方式も引き続き利用できます。

---

## ロードマップ

### P0 — Seed baseline

**完了。**

Seed-VCのP05 12秒rawを固定標準器として残します。Seed側へ追加のAdapterや補正を足しません。

### P1-BACKEND — 基盤モデル比較

Mosaic補正も個人適応も入れず、同一Source・同一Referenceで比較します。

```text
Seed-VC P05
vs
HQ-SVC P05
```

評価Sourceは中音域、高音ミックス、裏声、高速子音などへ分けます。HQ-SVCが有望な場合はSoulX-Singer-SVCを追加します。

### P2-REFERENCE — Reference検索

同一backend内で次を比較します。

```text
R0 固定P05
R1 Development set上のglobal best
R2 入力条件によるTop-1 retrieval
R3 Top-3生成＋人間Oracle
```

初期検索軸:

- target-relative register
- F0 span
- energy
- quality

R3が勝ち、R2が負ける場合は、Prompt Bankには価値があるがRetrieverの順位付けが弱いと判断します。

### P3-IDENTITY — Mosaicの中心仮説

固定backendのまま、低品質雑談由来のIdentity Memoryを導入します。

```text
M1 Acoustic retrievalのみ
M2 IdentityをPrompt検索に追加
M3 Identityを出力rerankにのみ追加
M4 検索とrerankの両方
```

最重要比較はM2/M3/M4対M1です。未知曲でも本人度が改善し、自然さが悪化しなければ中心仮説が成立します。

### P4-ADAPT — 必要な場合だけ

次の条件がそろった場合に限り検討します。

```text
自然さ      十分
発音保持    十分
F0・タイミング 十分
声区保持    十分
本人性だけ  不足
```

候補は小型speaker adapter、LoRA、timbre conditioning層などです。全面fine-tuningは最後の選択肢です。

---

## Quick Start

### 必要環境

Mosaic Core自体はPython 3.10以上と標準ライブラリだけで動きます。

追加ツール:

- 実際の音声生成: 各backend固有環境
- ブラインド音源のLUFS正規化: `ffmpeg`
- Seed-VC: 別リポジトリ・別仮想環境
- HQ-SVC: 当面はWSL/Linux側の別環境を想定

### インストールとテスト

```powershell
git clone https://github.com/MikanNigata/mosaic-svc-lab.git
cd mosaic-svc-lab

python -m pip install -e .
python -m unittest discover -s tests -v
```

### CLI

```powershell
mosaic-lab --help
```

---

## P1: Backend比較を準備する

設定例をコピーし、ローカルパスとbackend command templateを書き換えます。

```powershell
Copy-Item `
  configs/experiments/p1_hq_baseline.example.json `
  configs/experiments/p1_hq_baseline.local.json
```

まずdry-runで、実行予定のジョブとコマンドを確認します。

```powershell
mosaic-lab run `
  configs/experiments/p1_hq_baseline.local.json `
  --dry-run
```

実行:

```powershell
mosaic-lab run `
  configs/experiments/p1_hq_baseline.local.json `
  --fail-fast
```

実験ランナーは各ジョブについて次を記録します。

- experiment ID / condition ID
- backend
- source / reference
- random seed
- 実際の実行コマンド
- 入力ファイルのSHA-256
- 開始・終了時刻
- 所要時間
- return code
- stdout / stderr log
- canonical output path

結果は`manifest.jsonl`へ追記されます。

---

## ブラインド試聴セットを作る

```powershell
mosaic-lab blind `
  --manifest experiments/P1-HQ-001/manifest.jsonl `
  --output experiments/P1-HQ-001/listening `
  --normalize
```

`--normalize`を付けると、raw出力は保持したまま、試聴コピーだけをffmpegの2-pass loudness normalizationへ通します。

ブラインドファイル名と実条件の対応表はprivate mappingとして生成されます。対応表を見ずに試聴してください。

推奨評価軸:

- 本人度
- 自然さ
- 歌詞・発音保持
- F0・タイミング保持
- ビブラート・強弱保持
- 地声・ミックス・裏声の保持
- 音域間の本人性安定
- 金属音、ざらつき、二重声などのartifact
- 総合選好

---

## P2: PromptをTop-k検索する

現在のRetrieverは学習なしのPrototypeです。

```powershell
mosaic-lab rank `
  --source-features configs/retrieval/source_features.example.json `
  --prompt-index configs/retrieval/prompt_index.example.jsonl `
  --weights configs/retrieval/p2_retrieval.example.json `
  --top-k 3
```

現在の特徴:

```text
register_match
f0_span_match
energy_match
quality
```

初期重み:

```text
0.35 register
0.25 F0 span
0.20 energy
0.20 quality
```

絶対F0を直接一致させるのではなく、sourceとtargetそれぞれの音域内での相対位置を比較します。異なる性別や音域のsource/targetでも「高音フレーズにはtarget側の高音Reference」を選びやすくするためです。

この固定式は最終方式ではありません。まず、入力依存検索に改善余地があるかを判定するためのbaselineです。

---

## Backend Runnerの契約

Mosaic Coreはbackend内部を直接importしません。backend commandは最低限、次を受け取る薄いCLIへします。

```text
source path
reference path
output path
random seed
backend固有設定
```

概念的な実行例:

```text
mosaic-lab
  -> subprocess / wsl.exe
  -> backend専用CLI
  -> output.wav
  -> manifest.jsonl
```

HQ-SVCについては、公式環境の動作確認後に、WSL内で非対話推論できる薄いアダプターを追加する予定です。品質確認前にWindowsネイティブ移植や内部APIへの密結合は行いません。

---

## 実験設計の原則

### Development / Validation / Testを分離する

Promptや検索重みを選んだ曲で、最終性能を判定しません。

```text
Development
  Prompt候補、設定、検索重みを決める

Validation
  方式を選ぶ

Test
  最後に一度だけ評価する
```

### Source singerを複数にする

最低2〜3人のsourceを使い、source timbre leakageや音域依存を確認します。

### Promptと乱数を交差させる

diffusion系backendではPrompt差と乱数差が混ざる可能性があります。本格評価では各Promptを同じseed集合で生成します。

### 数値指標だけで順位を決めない

F0相関、cent RMSE、UV mismatchは破綻検出に使います。本人度、自然さ、声区保持、artifactはブラインド試聴を主評価にします。

---

## Go / No-Go

### 継続する条件

- 動的Top-1がglobal bestより未知曲で安定する
- Top-3内に人間が選ぶPromptが高頻度で入る
- Identity Memoryが本人度を改善する
- Identity導入で自然さが悪化しない
- データが少ない場合でも通常ゼロショットを下回らない
- データ量が増えるほど検索の安定性が上がる

### 設計変更する条件

- 常に同じPromptが勝つ
- Prompt検索順位と人間選好が無相関
- 雑談Identityが録音環境の類似度しか拾わない
- データ量を増やしても検索が改善しない
- 新backendが固定Seed baselineへ勝てない
- Top-3 Oracleでも固定Promptに勝てない

---

## リポジトリ構成

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

役割:

```text
mosaic-svc-lab
  実験定義、Memory仕様、Retriever、評価、ブラインド試聴

MikanNigata/seed-vc
  Seed baselineと過去のMosaic拡張コード

HQ-SVC / SoulX-Singer-SVC
  独立したbackend環境
```

---

## データ管理と責任ある利用

このリポジトリには、次をコミットしません。

- 元の音声データセット
- 長い生成WAV
- 個人用Speaker Memory
- model checkpoint / pretrained weight
- virtual environment
- privateなブラインド対応表

小さい比較用MP3だけを例外として保存できます。

音声変換は、本人の同意、著作権、肖像・パブリシティ、利用規約、地域の法令を確認したうえで使用してください。本リポジトリは、なりすまし、欺瞞、権利侵害を目的とした利用を意図していません。

**現時点でリポジトリ全体のライセンスは未設定です。** 公開されていること自体は、再配布・商用利用・派生物作成の許諾を意味しません。また、各backend、モデル重み、依存コードにはそれぞれ別のライセンスが適用されます。

---

## ドキュメント

- [Mosaic-SVC v2 Architecture](docs/architecture/MOSAIC_SVC_V2.md)
- [P1 / P2 Runbook](docs/experiments/P1_P2_RUNBOOK.md)
- [旧R1.6 Architecture](docs/architecture/MOSAIC_SVC_R16.md)
- [Seed Prompt Selection Experiments](docs/experiments/2026-07-31-prompt-selection.md)
- [P4-P8 Frozen Seed-VC Adaptation](docs/experiments/2026-08-02-p4-p8-adaptation.md)

---

## 現在の次タスク

1. P8採用版を未学習のフル曲で評価し、長時間・高音・語尾の破綻を確認する
2. F0 correlation、cent RMSE、UV errorを生成pipelineへ正式統合する
3. P8と固定Seed baselineをLUFS統一したブラインド試聴で比較する
4. 高品質歌唱Identity profileへ異話者negativeを追加し、本人度閾値を校正する
5. P8が主観評価でも勝った場合だけ、話者条件付きlossとStreaming Studentへ進む

現在の実用候補はP8です。

> Frozen Seed-VCへK/V-only LoRAとglobal Style-Slice Adapterを独立に学習し、両者を組み合わせて本人度と音質を同時に改善できるか。
