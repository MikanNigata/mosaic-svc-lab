# Mosaic-SVC Lab

日本語 | [English](README.en.md)

Mosaic-SVC Lab は、Seed-VC を基盤にした高品質 singing voice conversion の設計・実験用リポジトリです。

実用上の目的は単純です。

> 入力のガイド歌唱から音程・リズム・歌詞・歌い回しを保ちつつ、出力の声を対象歌手らしくする。

このリポジトリでは、研究設計、実験ログ、現在の最良設定、小さな聴き比べサンプル、再実行用スクリプトを管理します。実際に動くSeed-VC改造コードは別forkに置きます。

```text
runtime fork: https://github.com/MikanNigata/seed-vc
lab repo:     https://github.com/MikanNigata/mosaic-svc-lab
local root:   D:\voice-lab
```

## 背景

初期実験では、CAMPPlus/global style 経路へ小さなAdapterを足しても、聴感上は明確な改善になりませんでした。ブラインド比較では、数値上よかったAdapter版や長めprompt版ではなく、素のSeed-VC promptが選ばれました。

そのため、優先順位を変更します。

```text
変更前: speaker adapter / memory を先に強化
変更後: canonical prompt bank を先に作り、adapter は二段目
```

現在の作業仮説は次です。

> Seed-VCの歌唱変換では、対象話者らしさがreference prompt経路、とくにprompt melとprompt semanticに強く依存する。したがってprompt選別は単なる入力指定ではなく、話者資産として扱うべきである。

## Seed-VCの条件付け構造

44.1 kHzのSeed-VC歌唱モデルでは、参照音声から主に3つの条件が作られます。

```text
reference audio
  -> Whisper semantic prompt condition
  -> prompt mel
  -> CAMPPlus global style vector
```

入力歌唱を \(x\)、参照promptを \(r\)、style vectorを \(s\) とすると、現在の抽象化は次です。

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

各記号は次を意味します。

- \(C(x)\): Seed-VCのsource content特徴。
- \(F_0(x)\): 入力歌唱のピッチ軌跡。
- \(P_{\mathrm{sem}}(r)\): 参照音声から得たsemantic prompt条件。
- \(P_{\mathrm{mel}}(r)\): 参照音声のprompt mel。
- \(S_{\mathrm{camp}}(r)\): 192次元のCAMPPlus global style embedding。
- \(G_{\theta}\): 凍結したSeed-VCの音響生成器とvocoder。

最初に試したPrompt Adapterは、以下のような低ランク残差として実装しました。

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

ここで \(z_r\) はprompt summary、\(A,B\) は低ランク射影、\(\lambda\) は推論時のadapter強度です。

ただし現時点では、このAdapterよりもprompt選別そのものの方が聴感上重要そうです。

## 現在の最良設定

ブラインド試聴で選ばれた現在の設定です。

```yaml
condition: P05_12s_raw
prompt_source: dadadada_tenshi_vocal.wav
prompt_start_seconds: 48
prompt_duration_seconds: 12
diffusion_steps: 60
inference_cfg_rate: 0.50
adapter: none
```

ローカル再実行コマンド:

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

## 実験

### P0: Frozen Seed-VC Baseline

目的:

```text
Seed-VCのContent Encoder、Generator、Vocoderを変更せず、どこまで改善できるかを見る。
```

比較条件:

| ID | 条件 | 確認すること |
| --- | --- | --- |
| A | Seed-VC zero-shot | 基準性能 |
| B | 固定canonical prompt | prompt選別の効果 |
| C | Style Adapter | CAMPPlus/style経路の適応効果 |
| D0 | 推論時prototype補正 | CAMPPlus prototypeの効果 |
| M1 | Prompt bank選別 | prompt mel/semanticの効果 |
| M2 | Prompt Adapter | prompt経路への残差補正 |
| M4 | 雑談profileによるrerank | 低品質会話音声をidentity信号としてだけ使えるか |

### Prompt Bank実験

高品質な対象歌唱から12秒prompt候補を切り出しました。

重要な候補:

| Candidate | 区間 | メモ |
| --- | ---: | --- |
| P05 | 48s-60s | ブラインド試聴で勝った |
| P06 | 60s-72s | 以前は少しハキハキして聞こえた |
| P48_72 | 48s-72s | 数値上は強いが、試聴では勝たなかった |

### 雑談Speaker Profile

25分の雑談音声は音響学習には使っていません。

使ったのはCAMPPlusの話者重心だけです。

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

ここで \(K\) は外れ値を除いて残した雑談チャンク集合です。

prompt rerank scoreは次の形です。

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

\(\alpha=0.70\)、\(\beta=0.30\)、\(Q(r)\) は音質スコアです。

観測されたrerank:

| Rank | Prompt | CAMPPlus Similarity | Combined Score |
| ---: | --- | ---: | ---: |
| 1 | prompt_05_048.00s | 0.425142 | 0.594224 |
| 2 | prompt_07_072.00s | 0.423787 | 0.584651 |
| 3 | prompt_08_084.00s | 0.387137 | 0.554121 |
| 4 | prompt_03_024.00s | 0.483816 | 0.545296 |
| 5 | prompt_06_060.00s | 0.346047 | 0.529108 |

### ブラインド試聴

ブラインドセット:

```text
samples/blind_ittai40_p05_tests/
```

対応表:

| Blind File | 実際の条件 |
| --- | --- |
| test_01.mp3 | P48_72 24s raw |
| test_02.mp3 | P05 12s + Prompt Adapter strength 0.5 |
| test_03.mp3 | P05 12s raw |

選ばれたもの:

```text
test_03.mp3 -> P05 12s raw
```

### メトリクス

メトリクスはデバッグには有用でしたが、聴感の順位とは一致しませんでした。

| 条件 | F0 Corr | Cent RMSE | UV Mismatch | 試聴 |
| --- | ---: | ---: | ---: | --- |
| P05 12s raw | 0.968344 | 92.98 | 0.139872 | Preferred |
| P05 12s adapter strength 0.5 | 0.996016 | 48.94 | 0.123041 | 明確には勝たず |
| P05 12s adapter strength 1.0 | 0.994381 | 84.41 | 0.224898 | 強すぎ |
| P48_72 24s raw | 0.996700 | 44.93 | 0.073418 | 勝たず |

重要な負の結果:

```text
F0/UVメトリクスが良い = 本人らしさや自然さが良い、ではない。
```

## 現在の研究方針

次にやるべきことは、Adapterのランダムな調整ではなく、prompt bankの体系化です。

次の手順:

1. 高品質歌唱から20〜50個のprompt候補を切る。
2. 同じ評価クリップで一括生成する。
3. peakではなくLUFSで音量を揃える。
4. ブラインドで本人度と自然さを順位付けする。
5. 勝ったpromptの声区、energy、phonation、無音率、CAMPPlus類似度を分析する。
6. 安定したprompt bankができてから、その挙動をAdapterへ蒸留する。

## リポジトリ構成

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

## データ管理方針

このリポジトリには以下を入れません。

- 元データセット
- 長い生成WAV
- モデルcheckpoint
- pretrained weights
- virtual environment

実験確認に必要な小さいMP3サンプルだけは例外的に入れます。

## 現在の状態

```text
現時点で一番効いているのはprompt bank選別。
Prompt Adapterは実装済みだが、ブラインド試聴で勝つまでは二軍。
雑談データはspeaker profile信号としては使えるが、音響学習には使わない。
```
