@echo off
rem Launches the desktop trainer GUI with the local wpiper_venv.
cd /d "%~dp0"

if exist "wpiper_venv\Scripts\pythonw.exe" (
    start "" "wpiper_venv\Scripts\pythonw.exe" trainer_gui.py
) else (
    echo No venv found. Run setup_wpiper_gpu.bat or setup_wpiper_cpu.bat first.
    pause
)
