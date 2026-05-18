@echo off
REM ------------------------------------------------------------------------------------
REM Developed by Carpathian, LLC.
REM ------------------------------------------------------------------------------------
REM Legal Notice: Distribution Not Authorized.
REM ------------------------------------------------------------------------------------
REM Notes:
REM - Double-click to install dependencies (first run) and launch the Veritate
REM   dashboard. Re-runs skip already-satisfied steps and just relaunch.
REM start.bat
REM ------------------------------------------------------------------------------------

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY="
REM Prefer the newest Python in the supported range [3.10, 3.13] via py launcher.
REM Veritate's launcher does its own tier-aware version check; this just avoids
REM picking a too-new Python (e.g. 3.14) when an in-range one is also present.
where py >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    for %%V in (3.13 3.12 3.11 3.10) do (
        if "!PY!"=="" (
            py -%%V -c "import sys" >nul 2>&1
            if !ERRORLEVEL! EQU 0 set "PY=py -%%V"
        )
    )
    if "!PY!"=="" set "PY=py -3"
)
if "!PY!"=="" ( where python  >nul 2>&1 && set "PY=python"  )
if "!PY!"=="" ( where python3 >nul 2>&1 && set "PY=python3" )

if "!PY!"=="" (
    echo.
    echo Python 3.10+ is required but was not found on PATH.
    echo Install one of:
    echo   winget install Python.Python.3.12
    echo   https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

!PY! veritate.py %*
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo veritate exited with code %EXITCODE%.
    pause
)
exit /b %EXITCODE%
