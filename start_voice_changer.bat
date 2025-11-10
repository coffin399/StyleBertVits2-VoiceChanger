@echo off
setlocal enabledelayedexpansion

set "PY_VERSION=3.10"
set "VENV_DIR=.venv"
set "PROJECT_DIR=%~dp0"
set "SETUP_MARKER=%PROJECT_DIR%!VENV_DIR!\.deps_installed"

cd /d "%PROJECT_DIR%" || goto :EOF

where py >nul 2>&1
if errorlevel 1 (
    echo Python launcher 'py' not found. Please install Python %PY_VERSION% and ensure it is available via the 'py' command.
    goto :EOF
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment with Python %PY_VERSION%...
    py -%PY_VERSION% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment. Ensure Python %PY_VERSION% is installed.
        goto :EOF
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    goto :EOF
)

if not exist "%SETUP_MARKER%" (
    echo Installing project dependencies...
    python -m pip install --upgrade pip setuptools wheel
    if errorlevel 1 goto :deps_error
    python -m pip install -e .
    if errorlevel 1 goto :deps_error
    >"%SETUP_MARKER%" echo dependencies installed on %DATE% %TIME%
)

echo Launching StyleBertVits2 Voice Changer...
python -m app.main

goto :end

:deps_error
echo Failed to install dependencies. See the error message above.

:end
if defined VIRTUAL_ENV call deactivate
endlocal
