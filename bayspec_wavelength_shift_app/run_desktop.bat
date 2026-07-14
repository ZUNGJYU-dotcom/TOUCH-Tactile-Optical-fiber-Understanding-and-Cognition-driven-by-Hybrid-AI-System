@echo off
setlocal
cd /d "%~dp0"
set BAYSPEC_WAVELENGTH_APP_ROOT=%~dp0
if exist "D:\anaconda\miniconda3\python.exe" (
  "D:\anaconda\miniconda3\python.exe" desktop_launcher.py
) else if exist "E:\Codex\.venv_mfbq_alg\Scripts\python.exe" (
  "E:\Codex\.venv_mfbq_alg\Scripts\python.exe" desktop_launcher.py
) else (
  python desktop_launcher.py
)
