@echo off
rem Creates a venv next to this file and installs the CPU-only wpiper requirements.
rem Needs 64-bit Python 3.12 (tested). 3.11 should also work.
setlocal
cd /d "%~dp0"

set VENV=wpiper_venv
set PYCMD=python
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 --version >nul 2>nul
    if not errorlevel 1 ( set "PYCMD=py -3.12" ) else (
        py -3.11 --version >nul 2>nul
        if not errorlevel 1 set "PYCMD=py -3.11"
    )
)

echo Using: %PYCMD%
%PYCMD% --version || goto :fail

if exist "%VENV%\Scripts\python.exe" (
    echo Venv "%VENV%" already exists, reusing it.
) else (
    %PYCMD% -m venv "%VENV%" || goto :fail
)

call "%VENV%\Scripts\activate.bat" || goto :fail
python -m pip install --upgrade pip || goto :fail
pip install -r requirements_cpu.txt || goto :fail
pip install -e . --no-deps || goto :fail

python -c "import torch; print('torch', torch.__version__)"
python -c "from wpiper.phonemize_espeak import EspeakPhonemizer as E; print('espeak backend:', E().backend)"

echo.
echo Done. Put voices (name.onnx + name.onnx.json) in the voices folder, then run start_server.bat
echo or start the trainer with trainer_gui.bat. Activate the venv with:  %~dp0%VENV%\Scripts\activate.bat
pause
exit /b 0

:fail
echo.
echo Setup failed.
pause
exit /b 1
