import os
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def venv_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def main():
    base_dir = Path(__file__).resolve().parent
    req_file = base_dir / "requirements.txt"
    env_dir = base_dir / ".venv_tts"

    if not req_file.exists():
        print(f"requirements.txt not found at {req_file}")
        return 1

    # Always use venv - it's simple and works cross-platform
    if not env_dir.exists():
        print(f"Creating virtual environment: {env_dir}")
        run([sys.executable, "-m", "venv", str(env_dir)])
    else:
        print(f"Using existing environment: {env_dir}")

    python_exe = venv_python(env_dir)

    if not python_exe.exists():
        print(f"Python executable not found in venv: {python_exe}")
        return 1

    print("Upgrading pip...")
    run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], cwd=str(base_dir))

    print("Installing dependencies (this may take a few minutes due to llama-cpp-python compilation)...")
    run([str(python_exe), "-m", "pip", "install", "-r", str(req_file)], cwd=str(base_dir))

    if sys.platform.startswith("linux"):
        print("\n Ubuntu/Linux note: if soundfile import fails, install libsndfile1 with:")
        print("   sudo apt-get install libsndfile1")

    print("\n✓ Environment setup complete.")
    print("\nRun TTS with:")
    print(f"  {python_exe} tts.py --input data/truyen.txt --output data/truyen.wav --voice ly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
