import speech_recognition as sr
from faster_whisper import WhisperModel
import numpy as np


# Load Whisper model once (base model supports Vietnamese)
_model = None


def get_whisper_model(model_size="base"):
    """
    Get or load the Whisper model for offline Vietnamese STT.
    
    Args:
        model_size: "tiny", "base", "small", "medium", "large"
                   Larger = more accurate but slower
    
    Returns:
        WhisperModel instance
    """
    global _model
    if _model is None:
        print(f"Loading Whisper {model_size} model for offline STT...")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def listen_for_speech(timeout=10, phrase_time_limit=None):
    """
    Listen to microphone and convert speech to Vietnamese text (offline).
    
    Args:
        timeout: Max seconds to wait for speech
        phrase_time_limit: Max seconds per phrase
    
    Returns:
        Recognized text or empty string if failed
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.RequestError as e:
        print(f"Microphone error: {e}")
        return ""
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""

    # Convert audio to numpy array for Whisper
    try:
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
        
        model = get_whisper_model("base")
        segments, info = model.transcribe(audio_data, language="vi", beam_size=5)
        
        text = "".join([segment.text for segment in segments]).strip()
        if not text:
            print("Could not transcribe audio")
            return ""
        
        return text
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return ""


def listen_for_wakeword(wakeword="hey pi-bot", timeout=30):
    """
    Listen continuously for a wakeword, then record the prompt (offline).
    
    Args:
        wakeword: The trigger phrase to listen for
        timeout: Max seconds to wait for wakeword
    
    Returns:
        Recognized prompt text after wakeword, or empty string
    """
    print(f'Listening for "{wakeword}"...')
    recognizer = sr.Recognizer()

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
        except sr.RequestError as e:
            print(f"Microphone error: {e}")
            return ""
        except sr.UnknownValueError:
            print(f"Could not hear anything (attempt {attempt+1}/{max_attempts})")
            continue

        try:
            audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
            
            model = get_whisper_model("base")
            segments, info = model.transcribe(audio_data, language="vi", beam_size=5)
            
            text = "".join([segment.text for segment in segments]).strip()
            print(f"You said: {text}")

            if wakeword.lower() in text.lower():
                prompt = text.lower().replace(wakeword.lower(), "").strip()
                if prompt:
                    return prompt
                
                print('Wakeword detected! Now listening for your prompt...')
                print("Recording prompt (5 seconds)...")
                try:
                    with sr.Microphone() as source:
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    
                    audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
                    segments, info = model.transcribe(audio_data, language="vi", beam_size=5)
                    prompt_text = "".join([segment.text for segment in segments]).strip()
                    
                    return prompt_text
                except Exception as e:
                    print(f"Error recording prompt: {e}")
                    return ""
        except Exception as e:
            print(f"Whisper error (attempt {attempt+1}/{max_attempts}): {e}")

    print("Wakeword not detected after multiple attempts")
    return ""


def main():
    print("Testing offline Vietnamese speech recognition...")
    text = listen_for_speech()
    print(f"Recognized: {text}")


if __name__ == "__main__":
    main()

