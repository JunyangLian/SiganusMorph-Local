$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$V2Root = Resolve-Path (Join-Path $ScriptDir "..")
$TauriRoot = Join-Path $V2Root "desktop\tauri"
$CargoHome = "D:\1_postgraduate\.cargo"
$RustupHome = "D:\1_postgraduate\.rustup"
$MingwBin = "D:\1Bio_Soft\mingw64\bin"
$CargoTargetDir = "D:\1_postgraduate\tauri_target_siganusmorph_v2"

if (-not (Test-Path (Join-Path $TauriRoot "node_modules"))) {
    Write-Host "Tauri dependencies are not installed."
    Write-Host "Run npm install in siganusmorph_v2/desktop/tauri after explicit dependency-install approval."
    exit 1
}

if (Test-Path $CargoHome) {
    $env:CARGO_HOME = $CargoHome
    $env:PATH = (Join-Path $CargoHome "bin") + ";" + $env:PATH
}
if (Test-Path $RustupHome) {
    $env:RUSTUP_HOME = $RustupHome
}
if (Test-Path $MingwBin) {
    $env:PATH = $MingwBin + ";" + $env:PATH
}
if (-not (Test-Path $CargoTargetDir)) {
    New-Item -ItemType Directory -Force -Path $CargoTargetDir | Out-Null
}
$env:CARGO_TARGET_DIR = $CargoTargetDir

Set-Location $TauriRoot
Write-Host "Starting SiganusMorph V2.0 Tauri development shell..."
Write-Host "FastAPI backend should already be running at http://127.0.0.1:8000"
Write-Host "CARGO_TARGET_DIR=$env:CARGO_TARGET_DIR"
npm run tauri:dev
