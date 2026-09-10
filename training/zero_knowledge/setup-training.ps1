param(
  [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonCommand = $PythonPath
$PythonArguments = @()
if (-not $PythonCommand) {
  $PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
  if ($PythonLauncher) {
    $PythonCommand = $PythonLauncher.Source
    $PythonArguments = @("-3.12")
  } else {
    $PythonExecutable = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonExecutable) {
      throw "Python 3.12 was not found. Install it or pass -PythonPath."
    }
    $PythonCommand = $PythonExecutable.Source
  }
}
& $PythonCommand @PythonArguments -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"
if ($LASTEXITCODE -ne 0) {
  throw "The zero-knowledge environment requires Python 3.12."
}
$Venv = Join-Path $ProjectRoot ".venv-rl"
if (-not (Test-Path -LiteralPath (Join-Path $Venv "Scripts\python.exe"))) {
  & $PythonCommand @PythonArguments -m venv $Venv
}
$VenvPython = Join-Path $Venv "Scripts\python.exe"
& $VenvPython -m pip install numpy filelock typing-extensions sympy networkx jinja2 fsspec
& $VenvPython -c "import torch; assert torch.__version__ == '2.11.0+cu128'" 2>$null
if ($LASTEXITCODE -ne 0) {
  & $VenvPython -m pip install --no-deps "https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256=7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c"
}
& $VenvPython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
