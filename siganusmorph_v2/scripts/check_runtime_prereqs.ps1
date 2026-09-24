$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$V2Root = Resolve-Path (Join-Path $ScriptDir "..")
$ProjectRoot = Resolve-Path (Join-Path $V2Root "..")
$FrontendRoot = Join-Path $V2Root "frontend"
$TauriRoot = Join-Path $V2Root "desktop\tauri"
$CargoHome = "D:\1_postgraduate\.cargo"
$RustupHome = "D:\1_postgraduate\.rustup"
$MingwBin = "D:\1Bio_Soft\mingw64\bin"
$CargoTargetDir = "D:\1_postgraduate\tauri_target_siganusmorph_v2"

function Test-CommandAvailable {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Write-Status {
    param(
        [string]$Name,
        [bool]$Ok,
        [string]$Detail
    )
    $Status = if ($Ok) { "ok" } else { "missing" }
    Write-Host ("{0,-34} {1,-8} {2}" -f $Name, $Status, $Detail)
}

Set-Location $ProjectRoot

$PythonOk = Test-CommandAvailable "python"
$NodeOk = Test-CommandAvailable "node"
$NpmOk = Test-CommandAvailable "npm"
$CargoOk = (Test-CommandAvailable "cargo") -or (Test-Path (Join-Path $CargoHome "bin\cargo.exe"))
$RustcOk = (Test-CommandAvailable "rustc") -or (Test-Path (Join-Path $CargoHome "bin\rustc.exe"))
$GccOk = (Test-CommandAvailable "gcc") -or (Test-Path (Join-Path $MingwBin "gcc.exe"))
$FrontendDepsOk = Test-Path (Join-Path $FrontendRoot "node_modules")
$TauriDepsOk = Test-Path (Join-Path $TauriRoot "node_modules")

Write-Host "SiganusMorph Local V2.0 runtime prerequisite check"
Write-Host "Project root: $ProjectRoot"
Write-Host ""
Write-Status "Python command" $PythonOk "required for FastAPI backend and Admin Console"
Write-Status "Node.js command" $NodeOk "required for React/Vite frontend"
Write-Status "npm command" $NpmOk "required for dependency install and dev scripts"
Write-Status "Cargo command" $CargoOk "required for Tauri dev/build"
Write-Status "rustc command" $RustcOk "required for Tauri dev/build"
Write-Status "MinGW gcc command" $GccOk "required for GNU Windows Tauri toolchain"
Write-Status "Frontend node_modules" $FrontendDepsOk "required before npm run dev/build"
Write-Status "Tauri node_modules" $TauriDepsOk "required before npm run tauri:dev/build"
Write-Status "Cargo target dir" (Test-Path $CargoTargetDir) "recommended: $CargoTargetDir"

Write-Host ""
if (-not $FrontendDepsOk -or -not $TauriDepsOk) {
    Write-Host "Runtime status: blocked until npm dependencies are installed."
    Write-Host "Dependency installation requires explicit user approval in this environment."
    exit 0
}

Write-Host "Runtime status: dependencies appear installed."
exit 0
