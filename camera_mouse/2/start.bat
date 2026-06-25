@echo off
title Camera Mouse - Desktop Controller
color 0B

echo.
echo  ============================================================
echo              CAMERA MOUSE - Desktop Controller
echo       Control your real mouse with hand gestures!
echo  ============================================================
echo.
echo   1 Finger  =  Move cursor on whole screen
echo   2 Fingers =  Click
echo   3 Fingers =  Drag (move files/folders)
echo   4 Fingers =  Take Screenshot
echo   5 Fingers =  Pause / Stop
echo.
echo  ============================================================
echo.

cd /d "%~dp0"

:: Try to find Python
set PYTHON_CMD=

:: Check D:\Python\Python312 first (user's install location)
if exist "D:\Python\Python312\python.exe" (
    set PYTHON_CMD=D:\Python\Python312\python.exe
    goto :found
)

:: Check default python command
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :found
)

:: Check py launcher
py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    goto :found
)

echo  [ERROR] Python is not installed!
echo.
echo  Please install Python from: https://www.python.org/downloads/
echo  Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:found
echo  Using Python: %PYTHON_CMD%
echo.

echo  [1/2] Installing dependencies...
%PYTHON_CMD% -m pip install opencv-python mediapipe pyautogui "numpy<2" Pillow screeninfo --quiet
echo.

echo  [2/2] Starting Camera Mouse...
echo.
echo  ============================================================
echo   Press 'Q' on the camera window to quit
echo   Press 'R' to reset cursor to center
echo   Move mouse to screen corner = Emergency stop
echo  ============================================================
echo.

%PYTHON_CMD% camera_mouse.py

echo.
pause
