$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

Set-Location $ProjectRoot
Write-Host "Starting SiganusMorph V2.0 Admin Console..."
Write-Host "This console is for managers and developers, not the ordinary user entry."
streamlit run siganusmorph_v2/admin_streamlit/app.py
