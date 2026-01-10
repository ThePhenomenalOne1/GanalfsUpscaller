@echo off
echo ========================================
echo   AI Image Upscaler - Quick Setup
echo ========================================
echo.

REM Check for Python 3.9
py -3.9 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.9 is required but not found!
    echo Please install Python 3.9 from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment with Python 3.9...
py -3.9 -m venv venv

echo [2/4] Upgrading pip...
call venv\Scripts\pip install --upgrade pip

echo [3/4] Installing dependencies (this may take a few minutes)...
call venv\Scripts\pip install -r requirements.txt

echo [4/4] Downloading AI models...
call venv\Scripts\python upscaler.py --download-models

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To use the upscaler:
echo   1. Run GUI:     venv\Scripts\python gui.py
echo   2. Run CLI:     venv\Scripts\python upscaler.py image.jpg
echo.
pause
