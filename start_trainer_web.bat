@echo off
rem Starts the trainer web app (import, clean, prepare, train, monitor) at http://127.0.0.1:8100
cd /d "%~dp0"
if not exist "wpiper_venv\Scripts\python.exe" (
    echo No venv found. Run setup_wpiper_gpu.bat or setup_wpiper_cpu.bat first.
    pause
    exit /b 1
)
start "" http://127.0.0.1:8100
"wpiper_venv\Scripts\python.exe" trainer_server.py
pause
