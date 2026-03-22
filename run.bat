@echo off
REM takeout-rater launcher for Windows
REM Usage: run.bat  (from the repository root)

setlocal

REM Locate Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found.  Install Python 3.12+ from https://www.python.org/downloads/ >&2
    pause
    exit /b 1
)

REM Delegate to the cross-platform Python launcher
python "%~dp0scripts\launcher.py" %*
