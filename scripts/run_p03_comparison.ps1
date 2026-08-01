param(
  [string]$SeedVcRepo = "D:\voice-lab\seed-vc",
  [string]$Source = "D:\voice-lab\data\guide_vocals\ittai_itsukara_head_40s.wav",
  [string]$OutputRoot = "D:\voice-lab\out\mosaic_svc\p03_compare_ittai40",
  [int]$Steps = 60,
  [double]$Cfg = 0.50,
  [double]$PromptSeconds = 12.0,
  [double]$Lufs = -20.0
)

$ErrorActionPreference = "Stop"

$Python = Join-Path $SeedVcRepo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
  throw "Seed-VC python was not found: $Python"
}
if (-not (Test-Path -LiteralPath $Source)) {
  throw "Source file was not found: $Source"
}

$PromptP05 = "D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_05_048.00s.wav"
$PromptP06 = "D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_06_060.00s.wav"
$PromptP07 = "D:\voice-lab\out\mosaic_svc\p0\prompt_candidates\dadadada_12s\prompt_07_072.00s.wav"
$PromptAdapter = "D:\voice-lab\out\mosaic_svc\p0\prompt_adapter_p05_singing_only\prompt_adapter_final.pt"

$Conditions = @(
  [pscustomobject]@{
    Id = "P00_current_best_p05"
    Stage = "P00"
    Prompt = $PromptP05
    PromptAdapter = ""
    PromptAdapterStrength = ""
    Description = "Current best. P05 12s raw."
  },
  [pscustomobject]@{
    Id = "P01_quality_alt_p06"
    Stage = "P01"
    Prompt = $PromptP06
    PromptAdapter = ""
    PromptAdapterStrength = ""
    Description = "High-quality prompt-bank alternate. Tests prompt bank sensitivity."
  },
  [pscustomobject]@{
    Id = "P02_prompt_adapter_p05_s050"
    Stage = "P02"
    Prompt = $PromptP05
    PromptAdapter = $PromptAdapter
    PromptAdapterStrength = "0.50"
    Description = "P05 plus Prompt Adapter at strength 0.50."
  },
  [pscustomobject]@{
    Id = "P03_dialogue_rank1_p05"
    Stage = "P03"
    Prompt = $PromptP05
    PromptAdapter = ""
    PromptAdapterStrength = ""
    Description = "Dialogue-profile rerank rank 1. Same prompt as current best."
  },
  [pscustomobject]@{
    Id = "P03_dialogue_rank2_p07"
    Stage = "P03"
    Prompt = $PromptP07
    PromptAdapter = ""
    PromptAdapterStrength = ""
    Description = "Dialogue-profile rerank rank 2. First non-P05 rerank candidate."
  }
)

foreach ($condition in $Conditions) {
  if (-not (Test-Path -LiteralPath $condition.Prompt)) {
    throw "Prompt file was not found for $($condition.Id): $($condition.Prompt)"
  }
  if ($condition.PromptAdapter -and -not (Test-Path -LiteralPath $condition.PromptAdapter)) {
    throw "Prompt adapter file was not found for $($condition.Id): $($condition.PromptAdapter)"
  }
}

$RawRoot = Join-Path $OutputRoot "raw"
$NormRoot = Join-Path $OutputRoot "lufs_norm"
$Mp3Root = Join-Path $OutputRoot "mp3"
$LogRoot = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $RawRoot, $NormRoot, $Mp3Root, $LogRoot | Out-Null

function Invoke-Logged {
  param(
    [string[]]$Command,
    [string]$LogPath,
    [string]$WorkingDirectory
  )
  Add-Content -LiteralPath $LogPath -Encoding UTF8 -Value ("`n$ " + ($Command -join " "))
  Push-Location $WorkingDirectory
  $PreviousErrorActionPreference = $ErrorActionPreference
  try {
    # Native ML tools commonly write warnings/progress to stderr. Keep them in
    # the log, but only fail on the native process exit code.
    $ErrorActionPreference = "Continue"
    & $Command[0] $Command[1..($Command.Count - 1)] 2>&1 | ForEach-Object {
      Add-Content -LiteralPath $LogPath -Encoding UTF8 -Value $_.ToString()
    }
    $ExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    if ($ExitCode -ne 0) {
      throw "Command failed with exit code $ExitCode. See log: $LogPath"
    }
  }
  finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
    Pop-Location
  }
}

function Get-LatestWav {
  param([string]$Dir)
  $latest = Get-ChildItem -LiteralPath $Dir -Filter "*.wav" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (-not $latest) {
    throw "No wav output found in: $Dir"
  }
  return $latest.FullName
}

$Rows = @()
foreach ($condition in $Conditions) {
  $CondRaw = Join-Path $RawRoot $condition.Id
  New-Item -ItemType Directory -Force -Path $CondRaw | Out-Null
  $LogPath = Join-Path $LogRoot "$($condition.Id).log"

  $ArgsList = @(
    $Python,
    "-m", "mosaic_svc.p0.infer_p0",
    "--source", $Source,
    "--prompt", $condition.Prompt,
    "--output", $CondRaw,
    "--diffusion-steps", "$Steps",
    "--inference-cfg-rate", "$Cfg",
    "--prompt-seconds", "$PromptSeconds",
    "--f0-condition", "True",
    "--fp16", "True"
  )

  if ($condition.PromptAdapter) {
    $ArgsList += @(
      "--prompt-adapter", $condition.PromptAdapter,
      "--prompt-adapter-strength", $condition.PromptAdapterStrength
    )
  }

  Write-Host "Running $($condition.Id)"
  Invoke-Logged -Command $ArgsList -LogPath $LogPath -WorkingDirectory $SeedVcRepo

  $RawWav = Get-LatestWav $CondRaw
  $NormWav = Join-Path $NormRoot "$($condition.Id).wav"
  $Mp3 = Join-Path $Mp3Root "$($condition.Id).mp3"

  & ffmpeg -y -hide_banner -loglevel error -i $RawWav -af "loudnorm=I=$Lufs`:TP=-1.5:LRA=11" -ar 44100 -ac 2 $NormWav
  if ($LASTEXITCODE -ne 0) { throw "ffmpeg loudnorm failed for $($condition.Id)" }
  & ffmpeg -y -hide_banner -loglevel error -i $NormWav -codec:a libmp3lame -q:a 0 $Mp3
  if ($LASTEXITCODE -ne 0) { throw "ffmpeg mp3 encode failed for $($condition.Id)" }

  $Rows += [pscustomobject]@{
    id = $condition.Id
    stage = $condition.Stage
    description = $condition.Description
    prompt = $condition.Prompt
    prompt_adapter = $condition.PromptAdapter
    prompt_adapter_strength = $condition.PromptAdapterStrength
    raw_wav = $RawWav
    lufs_wav = $NormWav
    mp3 = $Mp3
    log = $LogPath
  }
}

$OutputsCsv = Join-Path $OutputRoot "p03_outputs.csv"
$Rows | Export-Csv -LiteralPath $OutputsCsv -Encoding UTF8 -NoTypeInformation

$EvalCsv = Join-Path $OutputRoot "p03_eval.csv"
$EvalArgs = @(
  $Python,
  "-m", "mosaic_svc.r16.eval_audio",
  "--reference", $Source,
  "--candidates"
) + ($Rows | ForEach-Object { $_.lufs_wav }) + @(
  "--output", $EvalCsv
)
Invoke-Logged -Command $EvalArgs -LogPath (Join-Path $LogRoot "eval.log") -WorkingDirectory $SeedVcRepo

$Readme = Join-Path $OutputRoot "README_RESULT.md"
$lines = @(
  "# Mosaic-SVC P03 comparison",
  "",
  "## Settings",
  "",
  "- source: $Source",
  "- steps: $Steps",
  "- cfg: $Cfg",
  "- prompt seconds: $PromptSeconds",
  "- f0-condition: true",
  "- fp16: true",
  "- lufs target: $Lufs",
  "",
  "## Outputs",
  ""
)
foreach ($row in $Rows) {
  $lines += "- $($row.id): $($row.mp3)"
}
$lines += @(
  "",
  "## Tables",
  "",
  "- $OutputsCsv",
  "- $EvalCsv",
  "",
  "## Interpretation",
  "",
  "- P00 and P03 rank1 are intentionally duplicated because dialogue rerank rank1 equals the current P05 prompt.",
  "- P03 rank2 is included to test whether dialogue-profile rerank provides a useful non-P05 alternative.",
  "- Numeric F0 metrics are do-no-harm checks only; listening preference remains the deciding signal."
)
Set-Content -LiteralPath $Readme -Encoding UTF8 -Value $lines

Write-Host "Saved outputs: $OutputsCsv"
Write-Host "Saved metrics: $EvalCsv"
Write-Host "Saved report: $Readme"
