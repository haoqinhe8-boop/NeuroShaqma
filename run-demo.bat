@echo off
rem NeuroShaqma one-click demo: auto-locate itself and auto-install deps.
rem If python is not in PATH, replace "python" below with the full path to python.exe
chcp 65001 >nul
echo [1/2] Checking numpy ...
python -c "import numpy" 2>nul
if errorlevel 1 (
    echo numpy not found, installing via pip ...
    pip install -r "%~dp0requirements.txt" 2>nul
    if errorlevel 1 pip install numpy
)
echo [2/2] Starting demo:
echo.
python -X utf8 "%~dp0demo.py"
pause
