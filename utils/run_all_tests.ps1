#!/usr/bin/env pwsh
<#
    Run the ADCTM FastAPI server and execute the endpoint‑testing script,
    all from inside the ADCTMEnv folder.

    Usage (from ADCTMEnv):
        .\run_all_tests.ps1

    Assumptions:
      • Virtual environment is located at ..\venv
      • Server entry point is ..\server\app.py
      • Test script is ADCTMEnv\test_server_endpoints.py
#>

# ------------------------------------------------------------
# 1️⃣ Activate the virtual environment (relative to ADCTMEnv)
# ------------------------------------------------------------
$venvActivate = Join-Path -Path $PSScriptRoot -ChildPath "..\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & $venvActivate
} else {
    Write-Error "Virtual‑env activation script not found at $venvActivate"
    exit 1
}

# ------------------------------------------------------------
# 2️⃣ Start the FastAPI server in the background
# ------------------------------------------------------------
$serverPath = Join-Path -Path $PSScriptRoot -ChildPath "..\server\app.py"
if (-not (Test-Path $serverPath)) {
    Write-Error "Server entry point not found at $serverPath"
    exit 1
}
Write-Host "Launching ADCTM server (uvicorn)..." -ForegroundColor Cyan
$serverProc = Start-Process -FilePath "python" -ArgumentList $serverPath -NoNewWindow -PassThru
# Give the server a moment to bind the port
Start-Sleep -Seconds 3

# ------------------------------------------------------------
# 3️⃣ Run the endpoint‑testing script (produces Rich table)
# ------------------------------------------------------------
$testScript = Join-Path -Path $PSScriptRoot -ChildPath "test_server_endpoints.py"
if (Test-Path $testScript) {
    Write-Host "Running endpoint tests..." -ForegroundColor Cyan
    python $testScript
} else {
    Write-Error "Test script not found at $testScript"
    $serverProc | Stop-Process
    exit 1
}

# ------------------------------------------------------------
# 4️⃣ Shut down the server
# ------------------------------------------------------------
Write-Host "Stopping ADCTM server..." -ForegroundColor Cyan
$serverProc | Stop-Process
Write-Host "All done!" -ForegroundColor Green
