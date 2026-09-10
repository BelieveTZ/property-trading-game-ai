param(
  [ValidateSet("background", "full")]
  [string]$Profile = "background",
  [string]$RunDir = "training/runs/zero-knowledge-main"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonPath = Join-Path $ProjectRoot ".venv-rl\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonPath)) {
  throw "Training environment is missing: $PythonPath"
}

$ResolvedRunDir = Join-Path $ProjectRoot $RunDir
New-Item -ItemType Directory -Force -Path $ResolvedRunDir | Out-Null
$StatusPath = Join-Path $ResolvedRunDir "status.json"
if (Test-Path -LiteralPath $StatusPath) {
  $Status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
  if ($Status.pid -and (Get-Process -Id $Status.pid -ErrorAction SilentlyContinue)) {
    throw "Training is already running (PID $($Status.pid))."
  }
}

$Common = @(
  "-m", "training.zero_knowledge.train",
  "--run-dir", $RunDir,
  "--players", "3,4,5",
  "--checkpoint-minutes", "5",
  "--resume"
)
if ($Profile -eq "background") {
  $TrainingArgs = $Common + @("--envs", "48", "--games-per-update", "48", "--minibatch", "768", "--cpu-threads", "6", "--snapshot-every", "2")
} else {
  $TrainingArgs = $Common + @("--envs", "96", "--games-per-update", "96", "--minibatch", "1024", "--cpu-threads", "12", "--snapshot-every", "2")
}

$Stdout = Join-Path $ResolvedRunDir "trainer.stdout.log"
$Stderr = Join-Path $ResolvedRunDir "trainer.stderr.log"
$Process = Start-Process -FilePath $PythonPath -ArgumentList $TrainingArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
$Process.PriorityClass = if ($Profile -eq "background") { "BelowNormal" } else { "Normal" }
Set-Content -LiteralPath (Join-Path $ResolvedRunDir "launcher-pid.txt") -Value $Process.Id -Encoding ascii
$Process.Id
