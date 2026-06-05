"""
Vietnamese Speech-to-Text (STT) Module
Specialized STT engine optimized for Vietnamese language using faster-whisper large-v3
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from faster_whisper import WhisperModel

# Run everything relative to this file
BASE_DIR = Path(__file__).resolve().parent


class VietnameseSpeechToText:
    """
    Specialized Speech-to-Text engine for Vietnamese language.
    Uses faster-whisper large-v3 model optimized for Vietnamese.
    """

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
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """
        Initialize Vietnamese STT engine.
        
        Args:
            device: Device to use ("cpu", "cuda", "auto")
            compute_type: Compute precision ("default", "int8", "int8_float16", "float16", "float32")
        """
        self.device = device
        self.compute_type = compute_type
        self.language = "vi"  # Vietnamese language code
        self.model_size = "large-v3"  # Best accuracy for Vietnamese
        self._model: Optional[WhisperModel] = None
    
    def _load_model(self) -> WhisperModel:
        """
        Lazy load the Whisper model on first use.
        
        Returns:
            WhisperModel instance
        """
        if self._model is None:
            print(f"📦 Loading faster-whisper {self.model_size} model for Vietnamese STT...")
            print(f"   Device: {self.device} | Compute: {self.compute_type}")
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as e:
                # If CUDA libs are unavailable (e.g. missing libcublas.so.12), keep STT functional on CPU.
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
            print(f"✓ Model loaded successfully\n")
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

    def record_microphone_audio(
        self,
        timeout: int = 10,
        phrase_time_limit: Optional[int] = 12,
        microphone_index: Optional[int] = None,
    ) -> np.ndarray:
        """Record audio from microphone and return float32 mono audio at 16kHz."""
        try:
            import speech_recognition as sr
        except ImportError:
            raise ImportError(
                "speech_recognition is required. Install with: pip install SpeechRecognition PyAudio"
            )

        recognizer = sr.Recognizer()
        if microphone_index is None:
            microphone_index = self._select_microphone_device_index(sr)

        print("🎤 Listening for Vietnamese speech...")
        if microphone_index is not None:
            print(f"🔊 Using microphone index: {microphone_index}")

        audio = self._capture_audio_with_fallback(
            recognizer,
            sr,
            timeout=timeout,
            phrase_time_limit=phrase_time_limit,
            microphone_index=microphone_index,
        )

        # Convert to Whisper-friendly format: 16kHz, 16-bit PCM mono, normalized float32.
        raw_pcm_16k = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_data = np.frombuffer(raw_pcm_16k, np.int16).astype(np.float32) / 32768.0
        print(f"✓ Audio captured ({len(audio_data)} samples at 16kHz)\n")
        return audio_data

    def play_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
        """Play mono float32 audio for quick microphone quality checks."""
        try:
            import pyaudio
        except ImportError:
            raise ImportError("PyAudio is required for playback. Install with: pip install PyAudio")

        pcm = np.clip(audio_data, -1.0, 1.0)
        pcm_bytes = (pcm * 32767.0).astype(np.int16).tobytes()

        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
            )
            stream.write(pcm_bytes)
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            pa.terminate()
    
    def transcribe_file(
        self,
        audio_path: str,
        beam_size: int = 5,
    ) -> str:
        """
        Transcribe Vietnamese audio from a file.
        
        Args:
            audio_path: Path to audio file
            beam_size: Beam search size (higher = more accurate but slower)
        
        Returns:
            Transcribed Vietnamese text
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"📄 Transcribing Vietnamese audio: {audio_path}")
        
        segments, info = self._transcribe_with_runtime_fallback(
            str(audio_path),
            language="vi",
            beam_size=beam_size,
        )
        
        text = "".join([segment.text for segment in segments]).strip()
        
        if not text:
            print("⚠️  No speech detected in audio")
            return ""
        
        print(f"✓ Transcription complete")
        return text
    
    def transcribe_microphone(
        self,
        timeout: int = 10,
        phrase_time_limit: Optional[int] = 12,
        beam_size: int = 5,
        microphone_index: Optional[int] = None,
    ) -> str:
        """
        Record and transcribe Vietnamese speech from microphone.
        
        Args:
            timeout: Max seconds to wait for input
            phrase_time_limit: Max seconds of captured speech per utterance
            beam_size: Beam search size
            microphone_index: Optional microphone device index (None = auto select)
        
        Returns:
            Transcribed Vietnamese text
        """
        try:
            import speech_recognition as sr
        except ImportError:
            raise ImportError(
                "speech_recognition is required. Install with: pip install SpeechRecognition PyAudio"
            )
        
        try:
            audio_data = self.record_microphone_audio(
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
                microphone_index=microphone_index,
            )
        except sr.RequestError as e:
            raise RuntimeError(f"Microphone error: {e}")
        except sr.UnknownValueError:
            print("⚠️  Could not understand audio")
            return ""
        except Exception as e:
            raise RuntimeError(f"Failed to record audio: {e}")
        
        try:
            print(f"📊 Processing audio ({len(audio_data)} samples)")
            
            segments, info = self._transcribe_with_runtime_fallback(
                audio_data,
                language="vi",
                beam_size=beam_size,
            )
            
            text = "".join([segment.text for segment in segments]).strip()
            
            if not text:
                print("⚠️  No speech detected")
                return ""
            
            print(f"✓ Transcription complete\n")
            return text
        except Exception as e:
            print(f"⚠️  Transcription failed: {e}")
            return ""
    
    def transcribe_with_confidence(
        self,
        audio_input,
        beam_size: int = 5,
    ) -> Tuple[str, float]:
        """
        Transcribe with confidence score.
        
        Args:
            audio_input: File path or numpy array
            beam_size: Beam search size
        
        Returns:
            Tuple of (text, confidence_score)
        """
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
        elif isinstance(audio_input, np.ndarray):
            audio_path = audio_input
        else:
            raise ValueError("audio_input must be a file path or numpy array")
        
        segments, info = self._transcribe_with_runtime_fallback(
            audio_path,
            language="vi",
            beam_size=beam_size,
        )
        
        text = "".join([segment.text for segment in segments]).strip()
        
        # Calculate average confidence
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


# Global instance
_global_vietnamese_stt: Optional[VietnameseSpeechToText] = None


def get_vietnamese_stt(
    device: str = "cpu",
    compute_type: str = "int8",
) -> VietnameseSpeechToText:
    """
    Get or create global Vietnamese STT instance.
    
    Args:
        device: Device to use
        compute_type: Compute precision
    
    Returns:
        VietnameseSpeechToText instance
    """
    global _global_vietnamese_stt
    
    if _global_vietnamese_stt is None:
        _global_vietnamese_stt = VietnameseSpeechToText(
            device=device,
            compute_type=compute_type,
        )
    
    return _global_vietnamese_stt


def transcribe_vietnamese_file(audio_path: str) -> str:
    """Quick Vietnamese file transcription."""
    stt = get_vietnamese_stt()
    return stt.transcribe_file(audio_path)


def transcribe_vietnamese_microphone(timeout: int = 10) -> str:
    """Quick Vietnamese microphone transcription."""
    stt = get_vietnamese_stt()
    return stt.transcribe_microphone(timeout=timeout, phrase_time_limit=12)


if __name__ == "__main__":
    print("=" * 60)
    print("Vietnamese Speech-to-Text Module")
    print("Powered by faster-whisper large-v3")
    print("=" * 60)
    print()
    
    stt = VietnameseSpeechToText()
    print("✓ Vietnamese STT module initialized and ready to use")
    print()
    print("Usage examples:")
    print("  from stt_vietnamese import get_vietnamese_stt")
    print("  stt = get_vietnamese_stt()")
    print("  text = stt.transcribe_file('audio.wav')")
    print("  text = stt.transcribe_microphone()")
    print("  text, confidence = stt.transcribe_with_confidence('audio.wav')")
