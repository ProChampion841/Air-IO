@echo off
REM Quick Start Script for Real-Time Pipeline Demo

echo ========================================
echo AirIO Real-Time Pipeline Demo
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo.
echo Running real-time pipeline demo...
echo This will take about 15-20 seconds...
echo.

python realtime_pipeline_demo.py

if errorlevel 1 (
    echo.
    echo ERROR: Demo failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Demo completed successfully!
echo ========================================
echo.
echo Results saved to: realtime_demo_results/
echo.

pause
