@echo off
rem Starts the TTS server (OpenAI-style /v1/audio/speech, dashboard at /, audiobook maker at /audiobook).
rem Voices are read from .\voices (name.onnx + name.onnx.json). Override with WPIPER_VOICES_DIR.
cd /d "%~dp0"
if not exist "wpiper_venv\Scripts\python.exe" (
    echo No venv found. Run setup_wpiper_gpu.bat or setup_wpiper_cpu.bat first.
    pause
    exit /b 1
)
if not exist voices mkdir voices
start "" http://127.0.0.1:8035/audiobook
"wpiper_venv\Scripts\python.exe" wpiper_server.py
pause
