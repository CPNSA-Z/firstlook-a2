# FirstLook inference service - start script for the GPU PC.
# Edit the settings below if needed, then right-click > Run with PowerShell.
# Default layout:  <folder>\inference_server\ (this script),  <folder>\Clean_Data\,  <folder>\models\
$DataDir   = Join-Path $PSScriptRoot "..\Clean_Data"   # df_cohort_final.pkl and sub-* folders
$ModelsDir = Join-Path $PSScriptRoot "..\models"       # Twostage_PRO_fold1..5_*.pth
# ADNI export (real units): next to this script, or in ..\metadata\
$MetaCsv   = Join-Path $PSScriptRoot "idaSearch_10_06_2025.csv"
if (-not (Test-Path $MetaCsv)) { $MetaCsv = Join-Path $PSScriptRoot "..\metadata\idaSearch_10_06_2025.csv" }
$Device    = ""          # "" = GPU if available (falls back to CPU on out-of-memory); "cpu" to force CPU
$Token     = "change-me"                    # the same value goes into the app's run script
# PyTorch build. RTX 50-series (Blackwell) needs CUDA 12.8 or newer: cu128.
$TorchIndex = "https://download.pytorch.org/whl/cu128"

# Do NOT use $ErrorActionPreference = "Stop" here: Windows PowerShell turns anything a native program
# writes to stderr (pip warnings, Python tracebacks, uvicorn logs) into a fatal error. Exit codes are
# checked instead, and Python's own messages are shown in full.
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot

function Stop-WithMessage($msg) {
    Write-Host ""
    Write-Host "ERROR: $msg" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

# 1) A real Python (the Microsoft Store 'python' stub does nothing)
$py = $null
foreach ($cand in @("py", "python", "python3")) {
    $cmd = Get-Command $cand -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notmatch "WindowsApps") {
        & $cand --version *> $null
        if ($LASTEXITCODE -eq 0) { $py = $cand; break }
    }
}
if (-not $py) { Stop-WithMessage "Python not found. Install Python 3.12 from python.org (tick 'Add python.exe to PATH')." }
Write-Host "Python: $(& $py --version 2>&1)"

# 2) Files
foreach ($d in @($DataDir, $ModelsDir)) { if (-not (Test-Path $d)) { Stop-WithMessage "Folder not found: $d" } }
if (-not (Test-Path (Join-Path $DataDir "df_cohort_final.pkl"))) { Stop-WithMessage "df_cohort_final.pkl not in $DataDir" }
$n = @(Get-ChildItem $ModelsDir -Filter "Twostage_PRO_fold*.pth").Count
if ($n -ne 5) { Stop-WithMessage "Expected 5 Twostage_PRO_fold*.pth files in $ModelsDir, found $n" }

# 3) PyTorch with CUDA, then the other packages
& $py -c "import torch" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyTorch ($TorchIndex) - about 3 GB, first run only..." -ForegroundColor Yellow
    & $py -m pip install torch --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) { Stop-WithMessage "PyTorch install failed (see messages above)." }
}
& $py -c "import fastapi, uvicorn, nibabel, scipy, sklearn, matplotlib, pandas" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing requirements..." -ForegroundColor Yellow
    & $py -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Stop-WithMessage "pip install failed (see messages above)." }
}
& $py -c "import torch; ok = torch.cuda.is_available(); print('PyTorch', torch.__version__, '| CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else 'running on CPU')"
if ($LASTEXITCODE -ne 0) { Stop-WithMessage "PyTorch does not load (see messages above)." }

# 4) Start
$env:FL_DATA_DIR = $DataDir; $env:FL_MODELS_DIR = $ModelsDir; $env:FL_TOKEN = $Token
if ($Device) { $env:FL_DEVICE = $Device }
if (Test-Path $MetaCsv) { $env:FL_META = $MetaCsv }
else { Write-Host "Note: $MetaCsv not found - patient table will use coded values" -ForegroundColor Yellow }
# LAN address = the adapter that has a default gateway (skips WSL / Hyper-V virtual adapters)
$ip = (Get-NetIPConfiguration -ErrorAction SilentlyContinue |
       Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq "Up" } |
       Select-Object -First 1).IPv4Address.IPAddress
Write-Host "Starting. The laptop should use http://$($ip):8008  (Ctrl+C to stop)" -ForegroundColor Green
& $py server.py
if ($LASTEXITCODE -ne 0) { Stop-WithMessage "The service stopped with an error (full message above)." }
Read-Host "Service stopped. Press Enter to close"
