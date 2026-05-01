@echo off
title FaceWarp DSP - Starting...

:: Change to the project directory (works regardless of where you double-click)
cd /d "%~dp0"

:: Try the project virtual environment first, fall back to system Python
if exist ".venv\Scripts\python.exe" (
    set PYTHON=".venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set PYTHON="venv\Scripts\python.exe"
) else (
    set PYTHON=python
)

echo.
echo  FaceWarp DSP Pipeline
echo  =====================
echo  Starting Flask server...
echo  The browser will open automatically at http://localhost:8081
echo.
echo  Close this window to stop the server.
echo.

%PYTHON% app.py

:: If Python exits (e.g. error), keep the window open so the user can read it
echo.
echo  Server stopped. Press any key to close.
pause >nul
