@echo off
REM Batch script to install Python dependencies for strepsuis-mdr
REM This script creates a virtual environment and installs all required packages

echo === StrepSuis-MDR Dependency Installation ===
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.8 or higher from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to:
    echo   1. Check 'Add Python to PATH' during installation
    echo   2. Restart Command Prompt after installation
    pause
    exit /b 1
)

echo Found Python:
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q .venv
)

python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)

echo Virtual environment created successfully!
echo.

REM Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies from requirements.txt
echo.
echo Installing dependencies from requirements.txt...
if exist requirements.txt (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies from requirements.txt!
        pause
        exit /b 1
    )
) else (
    echo WARNING: requirements.txt not found!
)

REM Install package in editable mode
echo.
echo Installing strepsuis-mdr package in editable mode...
pip install -e .
if errorlevel 1 (
    echo WARNING: Failed to install package in editable mode!
)

echo.
echo === Installation Complete! ===
echo.
echo To activate the virtual environment, run:
echo   .venv\Scripts\activate.bat
echo.
pause


