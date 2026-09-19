@echo off
rem FCTX MATTER STUDIO -- first-time setup on a fresh Windows PC.
rem Installs uv if it is missing, syncs Python and the dependencies, fetches
rem the hand model and runs the environment check.  Safe to run again.
setlocal
cd /d %~dp0

where uv >nul 2>nul
if errorlevel 1 (
    echo [setup] uv not found; installing it from astral.sh ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo [setup] could not install uv.  Install it by hand from https://docs.astral.sh/uv/ and run this again.
        goto :fail
    )
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo [setup] syncing Python 3.12 and the dependencies (first time: a few minutes) ...
uv sync --frozen
if errorlevel 1 goto :fail

echo [setup] fetching the hand landmark model ...
uv run python tools\download_models.py
if errorlevel 1 goto :fail

echo [setup] checking the GPU, OpenGL, the model and the camera ...
if exist venue.toml (
    uv run python -m fctx --config venue.toml --check
) else (
    uv run python -m fctx --check
    echo.
    echo [setup] no venue.toml yet.  Create one with:
    echo [setup]     uv run python -m fctx --dump-config ^> venue.toml
    echo [setup] then measure the camera with:
    echo [setup]     uv run python tools\calibrate.py --camera 0
)
if errorlevel 1 goto :fail

echo.
echo [setup] done.  Start with:  uv run python -m fctx        (or run_kiosk.bat for an exhibition)
if not defined FCTX_NOPAUSE pause
exit /b 0

:fail
echo.
echo [setup] setup did not finish; see the lines above.
if not defined FCTX_NOPAUSE pause
exit /b 1
