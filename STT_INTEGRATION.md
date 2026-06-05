# STT Module Integration Guide

## Files Created

### 1. **stt.py** (Main Module)
The core speech-to-text module with the `FasterWhisperSTT` class and convenience functions.

**Key Components:**
- `FasterWhisperSTT` class - Main STT engine
- `get_stt_engine()` - Global instance manager
- `transcribe_file()` - Quick file transcription
- `transcribe_microphone()` - Quick microphone transcription

### 2. **stt_examples.py** (Examples)
10 comprehensive usage examples demonstrating all features.

**Run examples:**
```bash
python stt_examples.py
```

### 3. **STT_MODULE.md** (Documentation)
Complete API reference and usage guide.

---

## Quick Integration into Existing Code

### Option 1: Replace Microphone Input
Update `ui_tts.py` to use the new STT module:

```python
# Before (current code):
from speech_input import listen_for_speech

# After (using new STT module):
from stt import get_stt_engine

stt = get_stt_engine(model_size="large-v3", language="vi")
user_text = stt.transcribe_from_microphone(timeout=10)
```

### Option 2: Create a Voice Input Interface
Create a Gradio interface with the STT module:

```python
import gradio as gr
from stt import get_stt_engine

stt = get_stt_engine()

def voice_input(audio):
    if audio is None:
        return ""
    return stt.transcribe(audio)

# Add to your Gradio interface:
voice_input_component = gr.Interface(
    fn=voice_input,
    inputs=gr.Audio(sources=["microphone", "upload"]),
    outputs="text"
)
```

### Option 3: Batch File Transcription
Transcribe all audio files in a directory:

```python
from pathlib import Path
from stt import get_stt_engine

stt = get_stt_engine(model_size="large-v3")

audio_dir = Path("data")  # Your audio directory
for audio_file in audio_dir.glob("*.wav"):
    text = stt.transcribe_from_file(str(audio_file))
    print(f"{audio_file.name}: {text}")
    
    # Optionally save transcription
    (audio_file.with_suffix(".txt")).write_text(text, encoding="utf-8")
```

---

## Model Selection Guide

For your project:

| Use Case | Recommended Model | Device | Compute Type |
|----------|-------------------|--------|--------------|
| Real-time UI input | `base` | auto | int8 |
| Batch processing (accuracy) | `large-v3` | auto | float16 |
| Low-resource environment | `tiny` | cpu | int8 |
| Maximum accuracy | `large-v3` | cuda | float16 |

---

## Performance Expectations

**First Run:**
- Model download: ~1-5 minutes (depending on internet)
- First transcription: ~30-60 seconds (model loading)

**Subsequent Runs:**
- Model reuse: Instant load from cache
- Transcription speed: Varies by hardware

**Speed Examples (5-second audio):**
- **tiny**: ~1 second (CPU)
- **base**: ~2-3 seconds (CPU)
- **large-v3**: ~5-10 seconds (CPU) / ~1-2 seconds (GPU)

---

## Troubleshooting

### Model Cache Location
- Linux/Mac: `~/.cache/huggingface/hub/`
- Windows: `%USERPROFILE%\.cache\huggingface\hub\`
- Takes ~3GB for large-v3 model

### GPU Support
```python
# Check if GPU is available:
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="auto")

# To force GPU:
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
```

### Memory Issues
If running on limited RAM:
```python
stt = FasterWhisperSTT(
    model_size="base",      # Smaller model
    device="cpu",           # Avoid GPU memory usage
    compute_type="int8"     # Quantized
)
```

---

## Next Steps

1. **Read** [STT_MODULE.md](STT_MODULE.md) for complete API reference
2. **Run** `stt_examples.py` to see all features
3. **Import** `from stt import get_stt_engine` in your code
4. **Update** existing speech recognition code to use the new module
5. **Optimize** model size/device for your hardware

---

## Dependencies Status

✅ All dependencies already in requirements.txt:
- `faster-whisper>=0.10.0`
- `SpeechRecognition>=3.10.0`
- `PyAudio>=0.2.14`

Just run:
```bash
pip install -r requirements.txt
```

---

## Module Architecture

```
stt.py
├── FasterWhisperSTT (Main class)
│   ├── transcribe() - Core method
│   ├── transcribe_from_file()
│   ├── transcribe_from_microphone()
│   ├── transcribe_with_confidence()
│   └── get_language_detection_result()
│
└── Global functions
    ├── get_stt_engine() - Singleton
    ├── transcribe_file() - Quick wrapper
    └── transcribe_microphone() - Quick wrapper
```

---

## Comparison with Existing Code

### Old `speech_input.py`:
- Uses "base" model
- Limited error handling
- Manual model management

### New `stt.py`:
- ✅ Uses "large-v3" model (more accurate)
- ✅ Comprehensive error handling
- ✅ Automatic model caching
- ✅ GPU support
- ✅ Confidence scoring
- ✅ Language detection
- ✅ Better documentation
- ✅ Easy integration

---

## Contributing & Extending

To add custom features:

```python
from stt import FasterWhisperSTT

class CustomSTT(FasterWhisperSTT):
    def transcribe_with_diarization(self, audio_input):
        """Add speaker diarization"""
        text, segments = self.transcribe(audio_input, return_segments=True)
        # Add custom processing here
        return text
```

---

## Questions?

See [STT_MODULE.md](STT_MODULE.md) for detailed API documentation or check the examples in `stt_examples.py`.
