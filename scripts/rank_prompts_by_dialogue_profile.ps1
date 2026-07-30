param(
  [string]$SeedVcRepo = "D:\voice-lab\seed-vc",
  [string]$Manifest = "D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_candidates.csv",
  [string]$Profile = "D:\voice-lab\out\mosaic_svc\speaker_profiles\maneki_dialogue25_campplus.pt",
  [string]$Output = "D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_ranked_by_dialogue_profile.csv"
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $SeedVcRepo ".venv\Scripts\python.exe"

Set-Location $SeedVcRepo
& $Python -m mosaic_svc.p0.rank_prompts_by_profile `
  --manifest $Manifest `
  --profile $Profile `
  --output $Output `
  --speaker-weight 0.70 `
  --quality-weight 0.30 `
  --fp16 True

