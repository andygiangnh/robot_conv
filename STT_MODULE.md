# Speech-to-Text (STT) Module - Documentation

## Overview

The STT module provides a clean, easy-to-use interface for speech-to-text transcription using **faster-whisper** with the **large-v3** model. It supports:

- ✅ **File transcription** - Convert audio files to text
- ✅ **Microphone input** - Real-time speech recognition
- ✅ **Multiple languages** - Vietnamese, English, Mandarin, and 90+ more
- ✅ **Confidence scoring** - Get confidence levels for transcriptions
- ✅ **Language detection** - Automatically detect the language of audio
- ✅ **GPU acceleration** - Automatic GPU detection and usage
- ✅ **Batch processing** - Transcribe multiple files efficiently
- ✅ **Error handling** - Comprehensive error messages and recovery

## Installation

The module requires `faster-whisper` which is already in `requirements.txt`:

```bash
pip install faster-whisper
```

For microphone input, also ensure these are installed:
```bash
pip install PyAudio SpeechRecognition
```

## Quick Start

### Basic Usage

```python
from stt import FasterWhisperSTT

# Initialize the STT engine
stt = FasterWhisperSTT(model_size="large-v3", language="vi")

# Transcribe from file
text = stt.transcribe_from_file("audio.wav")
print(text)

# Transcribe from microphone
text = stt.transcribe_from_microphone(timeout=10)
print(f"You said: {text}")
```

### Using Global Instance

```python
from stt import get_stt_engine

# Get or create global instance (reuses model for multiple calls)
stt = get_stt_engine(model_size="large-v3", language="vi")

text1 = stt.transcribe_from_file("audio1.wav")
text2 = stt.transcribe_from_file("audio2.wav")
```

### Quick Convenience Functions

```python
from stt import transcribe_file, transcribe_microphone

# File transcription
text = transcribe_file("audio.wav", language="vi", model_size="large-v3")

# Microphone transcription
text = transcribe_microphone(timeout=10, language="vi")
```

## API Reference

### Class: `FasterWhisperSTT`

Main class for speech-to-text functionality.

#### Constructor

```python
FasterWhisperSTT(
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "default",
    language: str = "vi"
)
```

**Parameters:**
- `model_size`: Model variant to use
  - "tiny" - Fastest, least accurate (~39M parameters)
  - "base" - Good balance (~74M parameters)
  - "small" - Better accuracy (~244M parameters)
  - "medium" - High accuracy (~769M parameters)
  - "large" - Very high accuracy (~1550M parameters)
  - "large-v3" - Latest & most accurate (~1550M parameters) [**Default**]

- `device`: Hardware to run on
  - "auto" - Auto-detect GPU, fallback to CPU [**Default**]
  - "cuda" - NVIDIA GPU (requires CUDA)
  - "cpu" - CPU only

- `compute_type`: Precision for computations
  - "default" - Auto-select optimal precision [**Default**]
  - "int8" - 8-bit quantization (fastest, lower memory)
  - "int8_float16" - Mixed precision
  - "float16" - 16-bit floating point
  - "float32" - 32-bit floating point (most accurate, slowest)

- `language`: Default language code
  - "vi" - Vietnamese
  - "en" - English
  - "zh" - Mandarin Chinese
  - (90+ language codes supported)

#### Methods

##### `transcribe(audio_input, language=None, beam_size=5, return_segments=False)`

Core transcription method supporting multiple input types.

**Parameters:**
- `audio_input`: Audio source (file path as str/Path or numpy array)
- `language`: Language code (uses default if None)
- `beam_size`: Decoding beam size (5-10 recommended, higher = more accurate but slower)
- `return_segments`: If True, returns segment objects along with text

**Returns:**
- Tuple of `(text: str, segments: List or None)`

**Example:**
```python
text, segments = stt.transcribe("audio.wav", language="vi", beam_size=5)
```

---

##### `transcribe_from_file(audio_path, language=None)`

Transcribe an audio file.

**Parameters:**
- `audio_path`: Path to audio file (supports .wav, .mp3, .flac, .ogg, etc.)
- `language`: Language code (uses default if None)

**Returns:**
- Transcribed text as string

**Example:**
```python
text = stt.transcribe_from_file("speech.wav", language="vi")
print(text)
```

---

##### `transcribe_from_microphone(timeout=10, language=None)`

Record and transcribe from microphone.

**Parameters:**
- `timeout`: Max seconds to wait for input before timeout
- `language`: Language code (uses default if None)

**Returns:**
- Transcribed text as string

**Raises:**
- `ImportError`: If SpeechRecognition or PyAudio not installed
- `RuntimeError`: If microphone access fails

**Example:**
```python
text = stt.transcribe_from_microphone(timeout=15, language="vi")
print(f"You said: {text}")
```

---

##### `transcribe_with_confidence(audio_input, language=None)`

Transcribe with confidence score.

**Parameters:**
- `audio_input`: Audio source (file path or numpy array)
- `language`: Language code (uses default if None)

**Returns:**
- Tuple of `(text: str, confidence: float)` where confidence is 0.0 to 1.0

**Example:**
```python
text, confidence = stt.transcribe_with_confidence("audio.wav")
print(f"Text: {text}")
print(f"Confidence: {confidence:.1%}")
```

---

##### `get_language_detection_result(audio_input)`

Detect the language of audio content.

**Parameters:**
- `audio_input`: Audio source (file path or numpy array)

**Returns:**
- Tuple of `(language_code: str, text: str, probability: float)`

**Example:**
```python
detected_lang, text, prob = stt.get_language_detection_result("audio.wav")
print(f"Language: {detected_lang} (confidence: {prob:.1%})")
print(f"Transcription: {text}")
```

---

### Module-Level Functions

#### `get_stt_engine(model_size="large-v3", device="auto", compute_type="default", language="vi")`

Get or create a global STT engine instance (singleton pattern for efficiency).

**Returns:**
- `FasterWhisperSTT` instance

**Example:**
```python
stt = get_stt_engine()  # Reuses same model for multiple calls
```

---

#### `transcribe_file(audio_path, language="vi", model_size="large-v3")`

Quick transcription from file using global instance.

**Example:**
```python
text = transcribe_file("audio.wav")
```

---

#### `transcribe_microphone(timeout=10, language="vi", model_size="large-v3")`

Quick transcription from microphone using global instance.

**Example:**
```python
text = transcribe_microphone(timeout=10)
```

---

## Advanced Usage

### Batch Transcription

```python
from pathlib import Path
from stt import get_stt_engine

stt = get_stt_engine()

# Transcribe all .wav files in a directory
audio_dir = Path("audio_files")
results = {}

for audio_file in audio_dir.glob("*.wav"):
    try:
        text = stt.transcribe_from_file(str(audio_file))
        results[audio_file.name] = text
        print(f"✓ {audio_file.name}")
    except Exception as e:
        print(f"✗ {audio_file.name}: {e}")

# Save results
import json
with open("transcriptions.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

### Working with NumPy Arrays

```python
import numpy as np
from stt import FasterWhisperSTT

stt = FasterWhisperSTT()

# Generate or load audio data as numpy array
# Audio should be float32 normalized to [-1, 1]
audio_array = np.random.randn(16000 * 5).astype(np.float32) / 32768.0

text = stt.transcribe(audio_array)
print(text)
```

### Multi-Language Processing

```python
from stt import FasterWhisperSTT

# Create separate instances for different languages
stt_vi = FasterWhisperSTT(language="vi")
stt_en = FasterWhisperSTT(language="en")
stt_zh = FasterWhisperSTT(language="zh")

# Transcribe language-specific files
vietnamese_text = stt_vi.transcribe_from_file("vietnamese_audio.wav")
english_text = stt_en.transcribe_from_file("english_audio.wav")
mandarin_text = stt_zh.transcribe_from_file("mandarin_audio.wav")
```

### Error Handling

```python
from stt import FasterWhisperSTT

stt = FasterWhisperSTT()

try:
    text = stt.transcribe_from_file("audio.wav")
except FileNotFoundError:
    print("Audio file not found")
except ValueError as e:
    print(f"Invalid audio input: {e}")
except RuntimeError as e:
    print(f"Transcription failed: {e}")
```

### Optimizing for Speed vs Accuracy

```python
from stt import FasterWhisperSTT

# Fast but less accurate (good for real-time)
stt_fast = FasterWhisperSTT(
    model_size="base",
    device="auto",
    compute_type="int8",
    language="vi"
)
text = stt_fast.transcribe_from_microphone(timeout=10)

# Accurate but slower (good for batch processing)
stt_accurate = FasterWhisperSTT(
    model_size="large-v3",
    device="cuda",
    compute_type="float16",
    language="vi"
)
text = stt_accurate.transcribe_from_file("audio.wav")
```

## Performance Tips

### 1. **GPU Acceleration**
```python
# Use GPU if available (much faster)
stt = FasterWhisperSTT(device="auto", compute_type="float16")
```

### 2. **Model Selection**
| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| tiny | 39M | ⚡⚡⚡ | ⭐ | Real-time, resource-constrained |
| base | 74M | ⚡⚡ | ⭐⭐ | Good balance |
| small | 244M | ⚡ | ⭐⭐⭐ | Better accuracy |
| medium | 769M | 🐢 | ⭐⭐⭐⭐ | High accuracy |
| large-v3 | 1550M | 🐢🐢 | ⭐⭐⭐⭐⭐ | Best accuracy |

### 3. **Quantization**
```python
# Smaller memory footprint, faster (slightly less accurate)
compute_type="int8"

# Better accuracy, more memory
compute_type="float16"
```

### 4. **Beam Size**
```python
# Faster transcription
beam_size=1

# Slower but more accurate
beam_size=10
```

## Supported Languages

The large-v3 model supports 90+ languages including:

- **Asian**: Vietnamese, Mandarin, Cantonese, Thai, Japanese, Korean
- **European**: English, French, German, Spanish, Portuguese, Italian, Polish, Dutch
- **Middle Eastern**: Arabic, Hebrew
- **And many more...**

Use language codes like: "vi", "en", "fr", "de", "es", "zh", "ja", "ko", etc.

## Audio Format Support

Works with various audio formats:
- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- OGG Vorbis (.ogg)
- AAC (.aac)
- And more via ffmpeg

## Troubleshooting

### Issue: "No module named 'faster_whisper'"
**Solution:** Install faster-whisper
```bash
pip install faster-whisper
```

### Issue: "No module named 'speech_recognition'" or PyAudio errors
**Solution:** Install dependencies for microphone input
```bash
pip install SpeechRecognition PyAudio
```

### Issue: "CUDA out of memory"
**Solution:** Use quantization or smaller model
```python
stt = FasterWhisperSTT(
    model_size="base",
    compute_type="int8",
    device="cuda"
)
```

### Issue: Slow transcription on CPU
**Solution:** Use GPU or smaller model
```python
stt = FasterWhisperSTT(device="cuda", compute_type="float16")
```

### Issue: Empty transcription
**Solution:** Check audio quality and noise levels
```python
text = stt.transcribe_from_file("audio.wav")
if not text.strip():
    print("Warning: No speech detected in audio")
```

## Integration with Other Modules

### With TTS (Text-to-Speech)

```python
from stt import get_stt_engine
from tts import synthesize_text

# Speech-to-Text
stt = get_stt_engine()
user_input = stt.transcribe_from_microphone(timeout=10)

# Do some processing...

# Text-to-Speech
synthesize_text(response_text, "output.wav", voice="ly")
```

### With UI (Gradio)

```python
import gradio as gr
from stt import get_stt_engine

stt = get_stt_engine()

def transcribe_audio(audio):
    if audio is None:
        return "No audio provided"
    return stt.transcribe(audio)

interface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(sources=["microphone", "upload"]),
    outputs="text"
)
interface.launch()
```

## Examples

See `stt_examples.py` for 10 comprehensive usage examples.

## References

- **faster-whisper**: https://github.com/SYSTRAN/faster-whisper
- **Whisper Model**: https://github.com/openai/whisper
- **Supported Languages**: Full list in faster-whisper documentation

## License

Same as the parent project.
