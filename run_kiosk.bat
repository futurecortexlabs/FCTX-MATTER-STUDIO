@echo off
rem FCTX MATTER STUDIO -- exhibition launcher.
rem Runs the studio in kiosk mode and restarts it if it ever exits.
rem Put your venue settings in venue.toml (start from: uv run python -m fctx --dump-config).
cd /d %~dp0
if not exist logs mkdir logs
set CONFIG=venue.toml
if not exist %CONFIG% (
    echo no %CONFIG% found; running with the defaults.  Create one with:
    echo     uv run python -m fctx --dump-config ^> %CONFIG%
    set CONFIG=
)
:loop
if "%CONFIG%"=="" (
    uv run python -m fctx --kiosk --log-file logs\fctx.log
) else (
    uv run python -m fctx --config %CONFIG% --kiosk --log-file logs\fctx.log
)
echo %date% %time%  exit code %errorlevel% >> logs\restarts.log
timeout /t 3 /nobreak >nul
goto loop
