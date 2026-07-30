param(
  [string]$SeedVcRepo = "D:\voice-lab\seed-vc",
  [string]$Source = "D:\voice-lab\data\guide_vocals\ittai_itsukara_head_40s.wav",
  [string]$Prompt = "D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_05_048.00s.wav",
  [string]$Output = "D:\voice-lab\out\mosaic_svc\p0\current_best_p05_raw",
  [int]$Steps = 60,
  [double]$Cfg = 0.50
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $SeedVcRepo ".venv\Scripts\python.exe"

Set-Location $SeedVcRepo
& $Python -m mosaic_svc.p0.infer_p0 `
  --source $Source `
  --prompt $Prompt `
  --output $Output `
  --diffusion-steps $Steps `
  --inference-cfg-rate $Cfg `
  --prompt-seconds 12 `
  --f0-condition True `
  --fp16 True

