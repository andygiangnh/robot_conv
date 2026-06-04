# Vietnamese TTS (Text-to-Speech) Application

A standalone Vietnamese text-to-speech converter powered by **Vieneu**, an advanced on-device Vietnamese TTS model with support for multiple voices and dialects.

## Features

- **Multiple Vietnamese Voices**: Support for 6 Vietnamese voices with different dialects (Northern, Central, Southern)
- **On-Device Processing**: No cloud dependency - runs completely offline
- **Fast Synthesis**: Optimized inference for quick audio generation
- **Batch Processing**: Convert entire text files to audio with automatic sentence splitting
- **Cross-Platform**: Works on Windows and Ubuntu 22+

## Program Description

This TTS app reads Vietnamese text from a file and converts it to natural-sounding audio in WAV format. It supports voice selection by name or partial matching and automatically handles sentence segmentation for better audio quality.

**Default Voices Available:**
- Bình (nam miền Bắc) - Male, Northern dialect
- Tuyên (nam miền Bắc) - Male, Northern dialect
- Vĩnh (nam miền Nam) - Male, Southern dialect
- Đoan (nữ miền Nam) - Female, Southern dialect
- **Ly (nữ miền Bắc)** - Female, Northern dialect ← Default
- Ngọc (nữ miền Bắc) - Female, Northern dialect

## Installation

### Windows

1. **Requirements**: Python 3.10 or higher

2. **Run the installer**:
   ```cmd
   install_tts_env.bat
   ```
   
   This will:
   - Create a Python virtual environment (`.venv_tts`)
   - Install all dependencies (numpy, soundfile, vieneu)
   - Display the run command when complete

3. **Wait for completion** (first run takes 5-10 minutes due to `llama-cpp-python` compilation)

### Linux / Ubuntu 22+

1. **Requirements**: Python 3.10+

2. **Install system dependency**:
   ```bash
   sudo apt-get update
   sudo apt-get install libsndfile1
   ```

3. **Run the installer**:
   ```bash
   chmod +x install_tts_env.sh
   ./install_tts_env.sh
   ```
   
   This will:
   - Create a Python virtual environment (`.venv_tts`)
   - Install all dependencies
   - Display the run command when complete

4. **Wait for completion** (first run takes 5-10 minutes)

## Usage

After installation, run the TTS app:

**Windows:**
```cmd
.venv_tts\Scripts\python.exe tts.py --input data\truyen.txt --output data\truyen.wav --voice ly
```

**Linux:**
```bash
.venv_tts/bin/python tts.py --input data/truyen.txt --output data/truyen.wav --voice ly
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `data/truyen.txt` | Input text file (relative to tts_app folder) |
| `--output` | `data/truyen.wav` | Output audio file (WAV format) |
| `--voice` | `ly` | Voice name or prefix (e.g., "ly", "binh", "tuyen") |

### Examples

```bash
# Use default voice (Ly)
python tts.py

# Use a specific voice
python tts.py --voice binh

# Custom input/output
python tts.py --input my_text.txt --output my_audio.wav --voice ngoc
```

## Folder Structure

```
tts_app/
├── tts.py                  # Main TTS application
├── requirements.txt        # Python dependencies
├── install_tts_env.bat     # Windows installer
├── install_tts_env.sh      # Linux installer
├── install_tts_env.py      # Cross-platform Python installer
├── run_tts.bat             # Windows run helper
├── run_tts.sh              # Linux run helper
├── data/                   # Input/output files
│   ├── truyen.txt          # Sample input text
│   └── *.wav               # Generated audio files
└── .venv_tts/              # Virtual environment (created after install)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'vieneu'"

**Solution**: Run the installer again. The environment may not have completed setup:
```bash
# Windows
install_tts_env.bat

# Linux
./install_tts_env.sh
```

### Issue: "ModuleNotFoundError: No module named 'soundfile'" on Linux

**Solution**: Install the system audio library:
```bash
sudo apt-get install libsndfile1
```

Then reinstall Python dependencies:
```bash
.venv_tts/bin/python -m pip install soundfile
```

### Issue: Installation is very slow / "Building wheel for llama-cpp-python"

**This is normal.** The first installation compiles `llama-cpp-python` (a C extension for LLM inference) from source. This takes 5-15 minutes depending on your CPU. Subsequent runs are fast.

**What to do**: Let it finish. Do not interrupt. If interrupted, delete `.venv_tts` folder and rerun the installer.

### Issue: "No such file or directory" for input file

**Solution**: Ensure your input text file is in the `data/` folder or use an absolute path:
```bash
# Relative (file must be in data/ folder)
python tts.py --input data/mytxt.txt

# Or absolute path works too
python tts.py --input /full/path/to/mytext.txt --output output.wav
```

### Issue: Permission denied on `install_tts_env.sh`

**Solution on Linux**: Make the script executable:
```bash
chmod +x install_tts_env.sh
./install_tts_env.sh
```

### Issue: Python launcher not found on Windows

**Solution**: Install Python 3.10+ from [python.org](https://www.python.org/downloads/) and ensure it's added to PATH. Verify with:
```cmd
py --version
```

---

## Why Venv Works (And Why Conda Failed)

### The Conda Issue
Earlier attempts to use Conda failed due to **PowerShell profile execution policy blocking**:

1. User's PowerShell has execution policies that prevent loading profile scripts
2. When `conda activate` or `conda run` is invoked, it tries to initialize the shell environment
3. This initialization loads the PowerShell profile (`profile.ps1`), which was blocked by security policy
4. The conda wrapper never got to execute the actual command
5. On Linux/Ubuntu, conda wrapper scripts worked, but added unnecessary complexity for this simple use case

### Why Venv Works Better

1. **No Shell Wrapper Needed**: Venv doesn't require shell initialization - it just creates a folder with a standalone Python executable
2. **Direct Execution**: The installer and app run Python directly via subprocess calls:
   ```python
   subprocess.check_call([str(python_exe), "-m", "pip", "install", "-r", req_file])
   ```
   No shell wrapper, no profile loading, no policy issues
3. **Cross-Platform Simplicity**: Same execution model on Windows, Linux, and macOS
4. **Self-Contained**: All dependencies are in `.venv_tts/` folder - portable and deletable
5. **No External Tools**: Doesn't depend on conda being installed or configured

### Performance
Both would have identical performance once set up. The choice of venv here was about **reliability and simplicity** rather than conda's package management features.

---

## Requirements

- **Python**: 3.10 or higher
- **OS**: Windows 10+ or Ubuntu 22+
- **Disk**: ~2GB for virtual environment + models
- **RAM**: 4GB minimum (8GB recommended)
- **Network**: Required for first install (downloads models from Hugging Face)

## License

This application uses:
- **Vieneu 2.7.0**: Apache 2.0 License
- **PyTorch**: BSD License
- Supporting libraries: See requirements.txt

## Support

For issues with Vieneu itself, visit: https://github.com/pnnbao97/VieNeu-TTS
