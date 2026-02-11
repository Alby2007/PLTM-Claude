# PLTM — One-line installer for Windows
# Usage: irm https://raw.githubusercontent.com/Alby2007/PLTM-Claude/main/install.ps1 | iex
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  PLTM - Persistent Long-Term Memory for Claude"
Write-Host "  Installing..."
Write-Host ""

# Check Python
$PY = $null
foreach ($cmd in @("python3.11", "python3", "python", "py -3.11", "py -3")) {
    try {
        $parts = $cmd -split " "
        $ver = & $parts[0] ($parts[1..$parts.Length] + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")) 2>$null
        if ($ver -match "^3\.(\d+)$" -and [int]$Matches[1] -ge 10) {
            $PY = $cmd
            Write-Host "  Using Python $ver ($cmd)"
            break
        }
    } catch {}
}

if (-not $PY) {
    Write-Host "  ERROR: Python 3.10+ not found."
    Write-Host "  Install: winget install Python.Python.3.11"
    Write-Host "  Or download from https://www.python.org/downloads/"
    exit 1
}

# Clone
$InstallDir = "$HOME\PLTM"
if (Test-Path $InstallDir) {
    Write-Host "  Updating existing install at $InstallDir..."
    Set-Location $InstallDir
    git pull --quiet
} else {
    Write-Host "  Cloning PLTM..."
    git clone --quiet https://github.com/Alby2007/PLTM-Claude.git $InstallDir
    Set-Location $InstallDir
}

# Run setup
$parts = $PY -split " "
& $parts[0] ($parts[1..$parts.Length] + @("setup_pltm.py")) 

Write-Host ""
Write-Host "  Done! Restart Claude Desktop to activate PLTM."
Write-Host ""
