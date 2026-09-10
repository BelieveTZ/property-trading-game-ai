param([string]$RunDir = "training/runs/zero-knowledge-main")

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonPath = Join-Path $ProjectRoot ".venv-rl\Scripts\python.exe"
& $PythonPath -m training.zero_knowledge.control pause --run-dir $RunDir
