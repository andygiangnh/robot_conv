@echo off
setlocal

REM Simple venv setup for Windows
set "ENV_DIR=.venv_tts"

where py >nul 2>&1
if errorlevel 1 (
  echo Python launcher [py] not found. Install Python 3.10+ and retry.
  exit /b 1
)

if not exist "%ENV_DIR%" (
  echo Creating virtual environment...
  py -m venv "%ENV_DIR%"
  if errorlevel 1 exit /b 1
)

echo Upgrading pip...
call "%ENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo Installing dependencies (may take minutes - llama-cpp-python compiles from source)...
call "%ENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo ✓ Setup complete. Run with:
echo   %ENV_DIR%\Scripts\python.exe tts.py --input data\truyen.txt --output data\truyen.wav --voice ly

endlocal
