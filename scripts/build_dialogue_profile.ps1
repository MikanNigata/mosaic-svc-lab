param(
  [string]$SeedVcRepo = "D:\voice-lab\seed-vc",
  [string]$InputAudio = "D:\voice-lab\out\dialogue\maneki_karaoke_stream\maneki_dialogue_strict_denoised.wav",
  [string]$Output = "D:\voice-lab\out\mosaic_svc\speaker_profiles\maneki_dialogue25_campplus.pt"
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $SeedVcRepo ".venv\Scripts\python.exe"

Set-Location $SeedVcRepo
& $Python -m mosaic_svc.p0.build_speaker_profile `
  --input $InputAudio `
  --output $Output `
  --max-segments 96 `
  --chunk-seconds 8 `
  --hop-seconds 16 `
  --keep-ratio 0.70 `
  --fp16 True

