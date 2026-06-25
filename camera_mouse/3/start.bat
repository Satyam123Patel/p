@echo off
title Camera Mouse - Desktop Controller
color 0B

echo.
echo  ============================================================
echo              CAMERA MOUSE - Desktop Controller
echo       Control your real mouse with hand gestures!
echo  ============================================================
echo.
echo   1 Finger   =  Move cursor anywhere
echo   2 Fingers  =  Click (one-shot, go back to 1F to click again)
echo   3 Fingers  =  Drag items (only after clicking with 2F)
echo   4 Fingers  =  Select text / content
echo   5 Fingers  =  Copy (Ctrl+C)
echo   6 Fingers  =  Paste (Ctrl+V)  [use both hands]
echo   7 Fingers  =  Take Screenshot  [use both hands]
echo   10 Fingers =  Show Help        [use both hands]
echo.
echo  ============================================================
echo.

cd /d "%~dp0"

:: Find Python
set PYTHON_CMD=

if exist "D:\Python\Python312\python.exe" (
    set PYTHON_CMD=D:\Python\Python312\python.exe
    goto :found
)

python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :found
)

py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    goto :found
)

echo  [ERROR] Python is not installed!
echo  Install from: https://www.python.org/downloads/
pause
exit /b 1

:found
echo  Using: %PYTHON_CMD%
echo.
echo  [1/2] Installing dependencies...
%PYTHON_CMD% -m pip install opencv-python mediapipe pyautogui "numpy<2" Pillow screeninfo --quiet
echo.
echo  [2/2] Starting Camera Mouse...
echo.
echo  Press 'Q' on camera window to quit
echo  Press 'R' to reset cursor to center
echo  Move mouse to screen corner = Emergency stop
echo  ============================================================
echo.

%PYTHON_CMD% camera_mouse.py

echo.
pause
