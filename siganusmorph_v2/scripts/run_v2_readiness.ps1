$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

Set-Location $ProjectRoot
Write-Host "Refreshing SiganusMorph V2.0 readiness report..."
python siganusmorph_v2/verify_v2_readiness.py
