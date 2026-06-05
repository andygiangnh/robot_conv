"""
Speech-to-Text (STT) module using faster-whisper with large-v3 model.
Supports transcription from microphone, audio files, and raw audio data.
Optimized for Vietnamese language with fallback support for other languages.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from faster_whisper import WhisperModel

# Run everything relative to this file
BASE_DIR = Path(__file__).resolve().parent


class FasterWhisperSTT:
    """
    Speech-to-Text engine using faster-whisper large-v3 model.
    Manages model loading and transcription with caching.
    """
    
    DEFAULT_MODEL_SIZE = "large-v3"
    DEFAULT_DEVICE = "auto"  # auto-detects GPU if available
    DEFAULT_COMPUTE_TYPE = "default"  # auto-selects optimal precision

    @staticmethod
    def _is_cuda_runtime_error(error: Exception) -> bool:
        """Return True when the error indicates missing CUDA runtime libraries."""
        msg = str(error).lower()
        cuda_markers = (
            "libcublas",
            "libcudnn",
            "cuda",
            "cudnn",
            "could not load library",
            "cannot be loaded",
            "ctranslate2",
        )
        return any(marker in msg for marker in cuda_markers)

    @staticmethod
    def _select_microphone_device_index(sr_module) -> Optional[int]:
        """Prefer real capture devices (USB/pulse/default) over HDMI pseudo-devices."""
        names = sr_module.Microphone.list_microphone_names()
        ranked_keywords = (
            "c-media",
            "usb audio",
            "usb",
            "mic",
            "microphone",
            "pulse",
            "default",
        )
        for keyword in ranked_keywords:
            for idx, name in enumerate(names):
                lname = name.lower()
                if keyword in lname and "hdmi" not in lname:
                    return idx
        return None

    def _capture_audio_with_fallback(self, recognizer, sr_module, timeout: int, phrase_time_limit: Optional[int], microphone_index: Optional[int]):
        """Capture microphone audio with fallback sample rates to avoid paInvalidSampleRate."""
        sample_rate_candidates = (16000, None, 48000, 44100)
        last_error: Optional[Exception] = None

        for sample_rate in sample_rate_candidates:
            try:
                sr_text = str(sample_rate) if sample_rate is not None else "default"
                print(f"🔧 Opening microphone (sample_rate={sr_text})")
                if sample_rate is None:
                    mic = sr_module.Microphone(device_index=microphone_index)
                else:
                    mic = sr_module.Microphone(device_index=microphone_index, sample_rate=sample_rate)

                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    return recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit,
                    )
            except Exception as e:
                last_error = e
                # PortAudio/ALSA can surface sample-rate/open failures with inconsistent
                # exception texts (including NoneType close errors). Keep trying fallbacks.
                continue

        if last_error is not None:
            raise RuntimeError(f"Microphone open failed after fallback attempts: {last_error}")
        raise RuntimeError("Microphone open failed with unknown error")
    
    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        language: str = "vi",
    ):
        """
        Initialize the STT engine.
        
        Args:
            model_size: Model size ("tiny", "base", "small", "medium", "large", "large-v3")
            device: Device to use ("cpu", "cuda", "auto")
            compute_type: Compute precision ("default", "int8", "int8_float16", "float16", "float32")
            language: Default language code (e.g., "vi" for Vietnamese)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model: Optional[WhisperModel] = None
    
    def _load_model(self) -> WhisperModel:
        """
        Lazy load the Whisper model on first use.
        
        Returns:
            WhisperModel instance
        """
        if self._model is None:
            print(f"📦 Loading faster-whisper {self.model_size} model...")
            print(f"   Device: {self.device} | Compute: {self.compute_type}")
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as e:
                # Common Linux issue: CUDA path detected but runtime libs (e.g. libcublas.so.12) are missing.
                # Retry on CPU so STT still works without GPU acceleration.
                if self.device in {"auto", "cuda"} and self._is_cuda_runtime_error(e):
                    print(f"⚠️  CUDA unavailable ({e})")
                    print("↪ Falling back to CPU mode (compute_type=int8)")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self._model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                    )
                else:
                    raise
            print(f"✓ Model loaded successfully")
        return self._model

    def _transcribe_with_runtime_fallback(self, audio_input, **kwargs):
        """Run transcription and retry once on CPU if CUDA runtime fails during inference."""
        model = self._load_model()
        try:
            return model.transcribe(audio_input, **kwargs)
        except Exception as e:
            if self.device in {"auto", "cuda"} and self._is_cuda_runtime_error(e):
                print(f"⚠️  CUDA runtime failed during transcription ({e})")
                print("↪ Retrying on CPU mode (compute_type=int8)")
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = None
                model = self._load_model()
                return model.transcribe(audio_input, **kwargs)
            raise
    
    def transcribe(
        self,
        audio_input,
        language: Optional[str] = None,
        beam_size: int = 5,
        return_segments: bool = False,
    ) -> Tuple[str, Optional[List]]:
        """
        Transcribe audio from file, microphone, or numpy array.
        
        Args:
            audio_input: Path to audio file or numpy array of audio samples
            language: Language code (e.g., "vi" for Vietnamese). If None, uses default.
            beam_size: Beam search size for decoding (higher = more accurate but slower)
            return_segments: If True, return segment objects along with text
        
        Returns:
            Tuple of (transcribed_text, segments_or_None)
        
        Raises:
            FileNotFoundError: If audio file path doesn't exist
            ValueError: If audio input is invalid
        """
        if language is None:
            language = self.language
        
        # Load audio from file if string path provided
        if isinstance(audio_input, str):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            print(f"📄 Transcribing: {audio_path}")
        elif isinstance(audio_input, Path):
            if not audio_input.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            audio_path = audio_input
            print(f"📄 Transcribing: {audio_path}")
        elif isinstance(audio_input, np.ndarray):
            audio_path = audio_input
            print(f"🎤 Transcribing audio array ({len(audio_input)} samples)")
        else:
            raise ValueError(
                "audio_input must be a file path (str/Path) or numpy array"
            )
        
        # Transcribe
        segments, info = self._transcribe_with_runtime_fallback(
            audio_path,
            language=language,
            beam_size=beam_size,
        )
        
        # Collect text from segments
        text = "".join([segment.text for segment in segments]).strip()
        
        if not text:
            print("⚠️  Warning: No speech detected or empty transcription")
        else:
            print(f"✓ Transcription complete ({len(text)} characters)")
        
        if return_segments:
            return text, segments
        else:
            return text, None
    
    def transcribe_from_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio from a file.
        
        Args:
            audio_path: Path to audio file
            language: Language code. If None, uses default.
        
        Returns:
            Transcribed text
        """
        text, _ = self.transcribe(audio_path, language=language)
        return text
    
    def transcribe_from_microphone(
        self,
        timeout: int = 10,
        phrase_time_limit: Optional[int] = 12,
        language: Optional[str] = None,
        microphone_index: Optional[int] = None,
    ) -> str:
        """
        Record audio from microphone and transcribe it.
        Requires PyAudio to be installed.
        
        Args:
            timeout: Max seconds to wait for input before timing out
            phrase_time_limit: Max seconds of captured speech per utterance
            language: Language code. If None, uses default.
            microphone_index: Optional microphone device index (None = auto select)
        
        Returns:
            Transcribed text
        
        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If microphone access fails
        """
        try:
            import speech_recognition as sr
        except ImportError:
            raise ImportError(
                "speech_recognition is required for microphone input. "
                "Install with: pip install SpeechRecognition PyAudio"
            )
        
        recognizer = sr.Recognizer()
        if microphone_index is None:
            microphone_index = self._select_microphone_device_index(sr)
        
        try:
            print("🎤 Listening for audio...")
            if microphone_index is not None:
                print(f"🔊 Using microphone index: {microphone_index}")
            audio = self._capture_audio_with_fallback(
                recognizer,
                sr,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
                microphone_index=microphone_index,
            )
            print("✓ Audio captured")
        except sr.RequestError as e:
            raise RuntimeError(f"Microphone error: {e}")
        except sr.UnknownValueError:
            print("⚠️  Could not understand audio")
            return ""
        except Exception as e:
            raise RuntimeError(f"Failed to record audio: {e}")
        
        # Convert to numpy array for faster-whisper
        try:
            # Force audio to Whisper-friendly format: 16kHz, 16-bit PCM mono.
            raw_pcm_16k = audio.get_raw_data(convert_rate=16000, convert_width=2)
            audio_data = np.frombuffer(raw_pcm_16k, np.int16).astype(np.float32) / 32768.0
            
            text, _ = self.transcribe(audio_data, language=language)
            return text
        except Exception as e:
            print(f"⚠️  Transcription failed: {e}")
            return ""
    
    def transcribe_with_confidence(
        self,
        audio_input,
        language: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Transcribe audio and return confidence score.
        
        Args:
            audio_input: Path to audio file or numpy array
            language: Language code. If None, uses default.
        
        Returns:
            Tuple of (transcribed_text, average_confidence_score)
        """
        text, segments = self.transcribe(
            audio_input, language=language, return_segments=True
        )
        
        if not segments:
            return text, 0.0
        
        # Calculate average confidence from segments
        confidence_scores = []
        for segment in segments:
            if hasattr(segment, 'confidence'):
                confidence_scores.append(segment.confidence)
        
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )
        
        return text, avg_confidence
    
    def get_language_detection_result(
        self,
        audio_input,
    ) -> Tuple[str, str, float]:
        """
        Detect the language of the audio input.
        
        Args:
            audio_input: Path to audio file or numpy array
        
        Returns:
            Tuple of (detected_language_code, transcribed_text, language_probability)
        """
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
        elif isinstance(audio_input, np.ndarray):
            audio_path = audio_input
        else:
            raise ValueError("audio_input must be a file path or numpy array")
        
        segments, info = self._transcribe_with_runtime_fallback(audio_path)
        text = "".join([segment.text for segment in segments]).strip()
        
        # info.language contains detected language code
        detected_lang = info.language if hasattr(info, 'language') else "unknown"
        detected_prob = (
            info.language_probability if hasattr(info, 'language_probability') else 0.0
        )
        
        return detected_lang, text, detected_prob


# Global instance for convenience
_global_stt: Optional[FasterWhisperSTT] = None


def get_stt_engine(
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "default",
    language: str = "vi",
) -> FasterWhisperSTT:
    """
    Get or create a global STT engine instance.
    
    Args:
        model_size: Model size to use
        device: Device to use
        compute_type: Compute precision
        language: Default language
    
    Returns:
        FasterWhisperSTT instance
    """
    global _global_stt
    
    if _global_stt is None:
        _global_stt = FasterWhisperSTT(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
        )
    
    return _global_stt


# Convenience functions using global instance
def transcribe_file(
    audio_path: str,
    language: str = "vi",
    model_size: str = "large-v3",
) -> str:
    """
    Quick transcription from audio file.
    
    Args:
        audio_path: Path to audio file
        language: Language code
        model_size: Model size to use
    
    Returns:
        Transcribed text
    """
    stt = get_stt_engine(model_size=model_size, language=language)
    return stt.transcribe_from_file(audio_path, language=language)


def transcribe_microphone(
    timeout: int = 10,
    language: str = "vi",
    model_size: str = "large-v3",
) -> str:
    """
    Quick transcription from microphone.
    
    Args:
        timeout: Max seconds to wait for input
        language: Language code
        model_size: Model size to use
    
    Returns:
        Transcribed text
    """
    stt = get_stt_engine(model_size=model_size, language=language)
    return stt.transcribe_from_microphone(
        timeout=timeout,
        phrase_time_limit=12,
        language=language,
    )


if __name__ == "__main__":
    # Example usage
    print("=== faster-whisper large-v3 STT Module ===\n")
    
    # Initialize STT engine
    stt = FasterWhisperSTT(
        model_size="large-v3",
        device="auto",  # Will use GPU if available
        language="vi",
    )
    
    print("✓ STT module ready for use")
    print("\nUsage examples:")
    print("  from stt import get_stt_engine")
    print("  stt = get_stt_engine()")
    print("  text = stt.transcribe_from_file('audio.wav')")
    print("  text = stt.transcribe_from_microphone()")
