#!/usr/bin/env pwsh
<#
    PowerShell script to launch the ADCTM FastAPI server, run the endpoint test script,
    and display the Rich‑formatted table.

    Usage (from the project root):
        .\ADCTMEnv\run_server_tests.ps1

    The script assumes a virtual environment exists at .\venv.
#>

# ------------------------------------------------------------
# 1️⃣ Activate the virtual environment
# ------------------------------------------------------------
$venvActivate = Join-Path -Path $PSScriptRoot -ChildPath "..\venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & $venvActivate
} else {
    Write-Error "Virtual environment activation script not found at $venvActivate"
    exit 1
}

# ------------------------------------------------------------
# 2️⃣ Start the FastAPI server in the background
# ------------------------------------------------------------
Write-Host "Starting ADCTM server (uvicorn)..." -ForegroundColor Cyan
$serverProc = Start-Process -FilePath "python" -ArgumentList "-m", "server.app" -NoNewWindow -PassThru
# Give the server a moment to bind the port
Start-Sleep -Seconds 3

# ------------------------------------------------------------
# 3️⃣ Run the endpoint‑testing script
# ------------------------------------------------------------
$testScript = Join-Path -Path $PSScriptRoot -ChildPath "test_server_endpoints.py"
if (Test-Path $testScript) {
    Write-Host "Running endpoint tests..." -ForegroundColor Cyan
    python $testScript
} else {
    Write-Error "Test script not found at $testScript"
    # Ensure we clean up the server before exiting
    $serverProc | Stop-Process
    exit 1
}

# ------------------------------------------------------------
# 4️⃣ Shut down the server
# ------------------------------------------------------------
Write-Host "Stopping ADCTM server..." -ForegroundColor Cyan
$serverProc | Stop-Process
Write-Host "All done!" -ForegroundColor Green
