# Mosaic Temporal Timbre Memory P0

## 1. 目的

TTM-P0は、高品質な対象歌唱を0.4秒程度の局所パッチへ分解し、入力歌唱の各時刻に近い対象パッチを時間連続性を考慮して検索できるかを検証する段階です。

この段階では検索結果をSeed-VC、DiT、mel、vocoderへ注入しません。目的は局所音色情報の存在と検索可能性を、生成品質と切り離して測定することです。

固定するSeed-VC条件は次です。

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

## 2. Key-Value Memory

1つの対象歌唱パッチを次の3要素として保存します。

- Key: relative register、F0 span、F0 slope、energy percentile、voiced ratio、quality
- Value: WAV、log-mel、MFCC、スペクトル要約、harmonic ratio、局所時系列
- Metadata: 元ファイル、開始・終了時刻、patch ID、抽出条件、採否理由

巨大な時系列をJSONへ書かず、`features/*.npz`へ圧縮保存します。`memory.jsonl`は検索と監査に必要な要約だけを持ちます。

## 3. Global Identityとの違い

CAMPPlusは発話全体の話者同一性を表すGlobal Identityとして有効ですが、0.4秒の局所音色Valueには使用しません。TTMは、同じ話者内の声区、F0変化、energy、倍音バランスの局所差を扱います。

```text
CAMPPlus Identity Memory: 誰の声か
Temporal Timbre Memory:   その時刻の発声状態に近い対象パッチはどれか
```

## 4. 特徴抽出

音声全体に対してF0、RMS、mel、MFCC、スペクトル特徴を一度だけ計算し、重複パッチへ集約します。パッチごとに`librosa.pyin`を再実行しないため、長尺音声でも計算量を抑えられます。

F0 slopeは、有声フレームの時刻を`t`、F0を`f_t`として次の線形回帰係数です。

```math
s_{F0}=\operatorname{slope}\left(t, 12\log_2 f_t\right)
```

単位はsemitones per secondです。有声フレームが不足する場合、`f0_valid=false`、slopeは`0.0`としてJSONを有限値に保ちます。

Relative registerは絶対Hzではなく、各音声自身の有声log-F0分布におけるパッチ中央値のpercentileです。p05からp95へclampして外れ値の影響を抑えます。入力と対象を絶対F0で直接比較しないため、男女差や原曲キー差に対して頑健です。

Energyも同様に、各音声自身のRMS dB分布内percentileを使用します。

## 5. Quality filtering

次を満たさないパッチは`accepted=false`として残します。

- active ratioが閾値未満
- clipping ratioが閾値超過
- 非有限サンプルを含む
- RMSが極端に低い
- F0 confidenceが閾値未満
- 有声フレーム不足
- パッチ解析失敗

1パッチの失敗でMemory全体を停止しません。入力不在、デコード失敗、音声がパッチ長より短い場合は致命的エラーとして停止します。

## 6. 出力形式

```text
temporal_memory/
├── memory.json
├── memory.jsonl
├── patches/
│   └── patch_000001.wav
└── features/
    └── patch_000001.npz
```

`memory.json`にはschema/feature version、SHA-256、作成時刻、Pythonと依存ライブラリのversion、解析条件、全体統計を保存します。

## 7. 検索スコア

初期距離は次です。

```math
d = 0.35d_r + 0.20d_{span} + 0.15d_{slope}
  + 0.15d_e + 0.05d_v - 0.10q
```

- `d_r`: relative register差
- `d_span`: F0 span差
- `d_slope`: F0 slope差
- `d_e`: energy percentile差
- `d_v`: voiced ratio差
- `q`: patch quality

欠損特徴は距離へ0として入れず、その重みを除外して残りを再正規化します。top-k内のsoft weightは次です。

```math
w_i = \frac{\exp(-(d_i-d_{min})/T)}{\sum_j \exp(-(d_j-d_{min})/T)}
```

## 8. 時間連続性

独立top-1ではなく、前時刻の選択パッチとのValue summary距離を加えます。

```math
C_i(t)=d_i(t)+\lambda_c d_{timbre}(i, i_{t-1})+J(i,i_{t-1})
```

`J`は対象歌唱上で離れたパッチへ飛ぶ場合のjump penaltyです。P0はgreedy selectorですが、`TemporalPathSelector` Protocolを介して将来Viterbiへ交換できます。

## 9. Confidence

Confidenceは0から1へ正規化し、次を組み合わせます。

- nearest distance
- top-1/top-2 margin
- softmax entropy
- queryの有効特徴率
- selected patch quality

`--min-confidence`未満では`selected_patch_id=null`を返します。候補がない場合も例外ではなく、selectionなし・confidence 0です。

## 10. CLI

```powershell
mosaic-lab temporal-enroll `
  --source D:\voice-lab\data\target_vocal.wav `
  --output D:\voice-lab\out\temporal_memory `
  --patch-seconds 0.40 `
  --hop-seconds 0.10
```

```powershell
mosaic-lab temporal-query `
  --source D:\voice-lab\data\source_vocal.wav `
  --memory D:\voice-lab\out\temporal_memory `
  --output D:\voice-lab\out\temporal_query.jsonl `
  --top-k 5 `
  --continuity-weight 0.25
```

```powershell
mosaic-lab temporal-visualize `
  --query D:\voice-lab\out\temporal_query.jsonl `
  --memory D:\voice-lab\out\temporal_memory `
  --output D:\voice-lab\out\temporal_report
```

出力先がディレクトリなら依存なしHTMLとsummary JSONを生成します。`.png`を指定する場合だけ`.[visualization]`が必要です。

## 11. 既知の制約

- 音素特徴は未実装で、schema上は`type=none`
- 局所Valueを生成器へ注入しない
- CAMPPlusをframe-level Valueとして使わない
- greedy smoothingであり、global optimumは保証しない
- 分離artifact、残響、伴奏漏れを専用モデルで判定しない
- absolute pitch一致を目的にしない
- GUI、リアルタイム、CUDA専用処理を含まない

## 12. 次フェーズ条件

Temporal Adapterを検討する前に、実歌唱で次を満たす必要があります。

- 高音Sourceが高音Target patchを安定して検索する
- 裏声Sourceが地声patchへ頻繁に誤検索しない
- patch切替が過剰ではない
- register別coverage不足をconfidenceで検出できる
- 固定promptより有用な局所情報がMemoryに存在する

満たさない場合はSeed-VCを改造せず、Key設計、声区推定、時間平滑化、Memory coverageを先に改善します。
