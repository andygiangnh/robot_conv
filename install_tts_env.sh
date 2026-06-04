#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$BASE_DIR/.venv_tts"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ and retry."
  exit 1
fi

if [ ! -d "$ENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$ENV_DIR"
fi

PYTHON_EXE="$ENV_DIR/bin/python"
if [ ! -x "$PYTHON_EXE" ]; then
  echo "Python executable not found in $ENV_DIR"
  exit 1
fi

echo "Upgrading pip..."
"$PYTHON_EXE" -m pip install --upgrade pip

echo "Installing dependencies (may take minutes - llama-cpp-python compiles from source)..."
"$PYTHON_EXE" -m pip install -r "$BASE_DIR/requirements.txt"

echo
if command -v apt-get >/dev/null 2>&1; then
  echo "Ubuntu note: if soundfile import fails, install libsndfile1 with:"
  echo "  sudo apt-get install libsndfile1"
fi

echo "✓ Setup complete. Run with:"
echo "  $PYTHON_EXE tts.py --input data/truyen.txt --output data/truyen.wav --voice ly"
