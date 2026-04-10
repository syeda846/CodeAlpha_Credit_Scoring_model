param(
  [ValidateSet("logistic", "decision_tree", "random_forest")]
  [string]$ModelType = "logistic",

  [string]$DataPath = "data/german_credit_data.csv",

  [switch]$CreateProxyTarget
)

$ErrorActionPreference = "Stop"

Write-Host "== Credit scoring model runner ==" -ForegroundColor Cyan
Write-Host "ModelType: $ModelType"
Write-Host "DataPath:  $DataPath"
Write-Host "Proxy:     $CreateProxyTarget"

if (-not (Test-Path "venv")) {
  Write-Host "Creating virtual environment (venv/)..." -ForegroundColor Yellow
  py -m venv venv
}

Write-Host "Activating venv..." -ForegroundColor Yellow
. .\venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements.txt

$argsList = @(
  "src/train_credit_model.py",
  "--data-path", $DataPath,
  "--model-type", $ModelType
)

if ($CreateProxyTarget) {
  $argsList += "--create-proxy-target"
}

Write-Host "Running training..." -ForegroundColor Yellow
python @argsList
