@echo off
echo ========================================
echo YOLO12 Training - Quick Start
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if YOLO is installed
python -c "import ultralytics" >nul 2>&1
if errorlevel 1 (
    echo Installing YOLO requirements...
    pip install -r requirements_yolo.txt
    if errorlevel 1 (
        echo Error: Failed to install YOLO requirements
        pause
        exit /b 1
    )
)

echo.
echo Starting YOLO12 training...
echo.

REM Run the quick start script
python quick_start_yolo.py

echo.
echo Press any key to exit...
pause >nul
