# R1.6 最初の実データ学習

## 目的

P11からP16までを合成スモークではなく、高品質対象歌唱で直列学習し、未学習曲に対するF0保持と実音声出力を確認する。

## データ

- 対象歌唱: `dadadada_tenshi_vocal.wav`、約104.7秒
- Train: 0–36秒、84–104秒、計約52秒
- Validation: 36–48秒、60–72秒、計24秒
- Test: 48–60秒、72–84秒、計24秒
- 低品質雑談25分は音響教師へ使用していない
- 単一曲しかないため、楽曲単位の対象話者test分離は未達。別曲ガイド3本を変換評価に使用した

## 学習結果

| Stage | Best step | Validation |
| --- | ---: | ---: |
| P11 Content Teacher | 500 | 0.065679 |
| P13 Causal Student | 1500 | 0.537989 |
| P14 Acoustic Converter | 2200 | 0.714661 |
| P15 AP Head | 1000 | 0.646415 |
| P15 NSF v2 | 3000 | 1.946642 |

P12 speaker leakage probeは実行していない。外部同条件の複数歌手データがなく、対象歌唱とJSUT朗読を話者分類するとdomain分類をspeaker leakageと誤認するためである。

## Checkpoint

本人音色を含むためcheckpoint本体は公開Releaseへ置かず、`D:\voice-lab\out\mosaic_svc\r16_real` に保存する。

| Stage | SHA-256 |
| --- | --- |
| P11 | `21E1827206EBD9B75F33866EA4EC46A0959AA45478ED2F996289A6B0DFEE4BD1` |
| P13 | `6EACAF014B535BBA21B4CA568F25BB395365B9DD94D5AD54318B057BCD87592F` |
| P14 | `B288E8C2610385ECF95B154FAE9EC78AA551FA2CF0E8EAE4FEF0CBBFE08DB0A1` |
| P15 AP | `075690F043E23BB9E2684BD05F2A703EB421B89DEA76B646781BC4AFAE356A2A` |
| P15 NSF v2 | `F4CB9B712F83398F3494605C17EE75817E09C39C4D72482E157859B82E4C9A55` |

## NSF v1失敗と修正

最初のNSFは有声音でもAP教師の平均noise ratioが約0.57となり、励振の半分以上が白色ノイズになった。R16出力はpYINで有声音を検出できず不合格だった。

修正:

- voiced noise excitationを推定APの15%へ制限
- unvoicedはnoise ratio 0.5以上を維持
- 無音cropでspectral convergenceの分母が発散しないよう下限を1.0へ変更
- NSFだけを再学習してv2を作成

正解mel/F0/APをNSF v1へ入れたoracle試験ではF0 correlation 0.944だったため、位相生成自体ではなく励振比と前段予測の問題と切り分けた。

## 未学習曲でのF0評価

| Clip | Model | F0 corr | Cent RMSE | UV mismatch |
| --- | --- | ---: | ---: | ---: |
| ittai mid | R16 v2 | 0.996738 | 61.56 | 0.1378 |
| ittai mid | P10 | 0.840260 | 139.99 | 0.2136 |
| ivy move | R16 v2 | 0.990207 | 69.37 | 0.1029 |
| ivy move | P10 | 0.996322 | 41.11 | 0.0944 |
| kuchizuke | R16 v2 | 0.975105 | 77.81 | 0.0937 |
| kuchizuke | P10 | 0.938513 | 126.46 | 0.0619 |

R16は3本ともF0 correlation 0.975以上。IvyではP10の方がcent/UVで良く、全面勝利ではない。

## 現在の判定

No-Go。

- P11–P16の実データ学習と未学習曲出力は技術的には完走した
- v1の完全な無声音化はv2で解消し、F0指標は改善した
- しかし人間評価では「聴けたものではない」音質で、実用条件を明確に不合格とした
- F0 correlationは音程軌跡しか測らず、音素、音色、明瞭度、金属感、自然さを保証しない
- 約52秒からスクラッチ学習したL1中心のAcoustic Converterはmelを過平滑化しやすい
- 簡易NSFはmulti-resolution STFTだけで学習しており、実用vocoderに必要な敵対学習・feature matching・十分な多話者事前学習がない
- R16音声は公開Releaseの通常試聴候補から削除し、P10を現行defaultとして維持する

P8/P10試聴: [GitHub Release](https://github.com/MikanNigata/mosaic-svc-lab/releases/tag/listening-check-2026-08-02)

R16音声は失敗資料としてローカルに保持するが、通常候補としては公開しない。
