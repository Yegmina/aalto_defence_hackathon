@echo off
echo ========================================
echo YOLO Data Augmentation - Easy Start
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Install requirements if needed
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Starting data augmentation...
echo.

REM Run the easy start script
python easy_start.py

echo.
echo Press any key to exit...
pause >nul


