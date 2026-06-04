@echo off
setlocal

set "ENV_DIR=.venv_tts"
if not exist "%ENV_DIR%\Scripts\activate.bat" (
  echo Environment not found. Run install_tts_env.bat first.
  exit /b 1
)

call "%ENV_DIR%\Scripts\activate.bat"
python tts.py %*

endlocal
