@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1

cd /d "%~dp0"

set "PYEXE=%~dp0..\ .venv\Scripts\python.exe"
set "PYEXE=%PYEXE: =%"
if exist "%PYEXE%" (
  echo [info] using venv python: %PYEXE%
) else (
  set "PYEXE=python"
  echo [warn] venv python not found, using: %PYEXE%
)

%PYEXE% run_train.py %*
echo.
pause

