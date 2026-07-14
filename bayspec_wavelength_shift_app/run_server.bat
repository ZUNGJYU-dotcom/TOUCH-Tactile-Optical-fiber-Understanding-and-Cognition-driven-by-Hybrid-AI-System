@echo off
setlocal
cd /d "%~dp0"
set BAYSPEC_WAVELENGTH_APP_ROOT=%~dp0
if exist "D:\anaconda\miniconda3\python.exe" (
  "D:\anaconda\miniconda3\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8640
) else if exist "E:\Codex\.venv_mfbq_alg\Scripts\python.exe" (
  "E:\Codex\.venv_mfbq_alg\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8640
) else (
  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8640
)
