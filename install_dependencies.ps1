# PowerShell script to install Python dependencies for strepsuis-mdr
# This script creates a virtual environment and installs all required packages

Write-Host "=== StrepSuis-MDR Dependency Installation ===" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
$pythonCmd = $null
$pythonPaths = @(
    "python",
    "python3",
    "py",
    "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe",
    "C:\Program Files\Python*\python.exe",
    "C:\Python*\python.exe"
)

foreach ($path in $pythonPaths) {
    try {
        if ($path -like "*\*") {
            # Handle wildcard paths
            $found = Get-ChildItem -Path $path -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found) {
                $pythonCmd = $found.FullName
                break
            }
        } else {
            $result = & $path --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $pythonCmd = $path
                break
            }
        }
    } catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.8 or higher from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Make sure to:" -ForegroundColor Yellow
    Write-Host "  1. Check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    Write-Host "  2. Restart PowerShell after installation" -ForegroundColor Yellow
    exit 1
}

Write-Host "Found Python: $pythonCmd" -ForegroundColor Green
$version = & $pythonCmd --version
Write-Host "Version: $version" -ForegroundColor Green
Write-Host ""

# Check Python version (must be 3.8+)
$versionOutput = & $pythonCmd --version 2>&1
$versionMatch = $versionOutput -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
        Write-Host "ERROR: Python 3.8 or higher is required!" -ForegroundColor Red
        Write-Host "Current version: $versionOutput" -ForegroundColor Red
        exit 1
    }
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists. Removing old one..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

& $pythonCmd -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment!" -ForegroundColor Red
    exit 1
}

Write-Host "Virtual environment created successfully!" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
$venvPython = ".venv\Scripts\python.exe"
$venvPip = ".venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: Virtual environment activation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $venvPip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to upgrade pip, continuing anyway..." -ForegroundColor Yellow
}
Write-Host ""

# Install dependencies from requirements.txt
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
if (Test-Path "requirements.txt") {
    & $venvPip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install dependencies from requirements.txt!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "WARNING: requirements.txt not found!" -ForegroundColor Yellow
}

# Install package in editable mode
Write-Host ""
Write-Host "Installing strepsuis-mdr package in editable mode..." -ForegroundColor Cyan
& $venvPip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to install package in editable mode!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Installation Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment, run:" -ForegroundColor Cyan
Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Or on Windows CMD:" -ForegroundColor Cyan
Write-Host "  .venv\Scripts\activate.bat" -ForegroundColor White
Write-Host ""


