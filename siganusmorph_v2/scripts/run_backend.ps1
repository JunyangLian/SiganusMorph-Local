$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

Set-Location $ProjectRoot
Write-Host "Starting SiganusMorph V2.0 FastAPI backend..."
Write-Host "URL: http://127.0.0.1:8000/api/health"
python -m uvicorn siganusmorph_v2.backend.app:app --host 127.0.0.1 --port 8000 --reload
