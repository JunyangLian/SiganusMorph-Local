$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$V2Root = Resolve-Path (Join-Path $ScriptDir "..")
$FrontendRoot = Join-Path $V2Root "frontend"

if (-not (Test-Path (Join-Path $FrontendRoot "node_modules"))) {
    Write-Host "Frontend dependencies are not installed."
    Write-Host "Run npm install in siganusmorph_v2/frontend after explicit dependency-install approval."
    exit 1
}

Set-Location $FrontendRoot
Write-Host "Starting SiganusMorph V2.0 React/Vite frontend..."
Write-Host "URL: http://127.0.0.1:5173"
npm run dev
