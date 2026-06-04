#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$BASE_DIR/.venv_tts"
PYTHON_EXE="$ENV_DIR/bin/python"

if [ ! -x "$PYTHON_EXE" ]; then
  echo "Environment not found. Run install_tts_env.sh first."
  exit 1
fi

cd "$BASE_DIR"
exec "$PYTHON_EXE" tts.py "$@"
