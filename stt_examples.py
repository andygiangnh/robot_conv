"""
Speech-to-Text Usage Examples
Demonstrates how to use the faster-whisper large-v3 STT module
"""

from stt import FasterWhisperSTT, get_stt_engine, transcribe_file, transcribe_microphone


def example_1_basic_file_transcription():
    """Example 1: Basic file transcription"""
    print("\n=== Example 1: Basic File Transcription ===")
    
    audio_file = "path/to/your/audio.wav"
    
    stt = FasterWhisperSTT(model_size="large-v3", language="vi")
    text = stt.transcribe_from_file(audio_file)
    
    print(f"Transcribed text: {text}")


def example_2_microphone_input():
    """Example 2: Microphone input with timeout"""
    print("\n=== Example 2: Microphone Input ===")
    
    stt = FasterWhisperSTT(model_size="large-v3", device="auto", language="vi")
    text = stt.transcribe_from_microphone(timeout=15)
    
    print(f"You said: {text}")


def example_3_with_confidence():
    """Example 3: Get transcription with confidence score"""
    print("\n=== Example 3: Transcription with Confidence ===")
    
    stt = FasterWhisperSTT(model_size="large-v3")
    audio_file = "path/to/your/audio.wav"
    
    text, confidence = stt.transcribe_with_confidence(audio_file)
    
    print(f"Text: {text}")
    print(f"Confidence: {confidence:.2%}")


def example_4_language_detection():
    """Example 4: Detect language of audio"""
    print("\n=== Example 4: Language Detection ===")
    
    stt = FasterWhisperSTT(model_size="large-v3")
    audio_file = "path/to/your/audio.wav"
    
    detected_lang, text, probability = stt.get_language_detection_result(audio_file)
    
    print(f"Detected language: {detected_lang}")
    print(f"Language probability: {probability:.2%}")
    print(f"Transcription: {text}")


def example_5_using_global_instance():
    """Example 5: Using global instance for convenience"""
    print("\n=== Example 5: Global Instance ===")
    
    # Get or create global instance
    stt = get_stt_engine(model_size="large-v3", language="vi")
    
    # Multiple calls reuse the same model
    text1 = stt.transcribe_from_file("audio1.wav")
    text2 = stt.transcribe_from_file("audio2.wav")
    
    print(f"File 1: {text1}")
    print(f"File 2: {text2}")


def example_6_quick_convenience_functions():
    """Example 6: Using quick convenience functions"""
    print("\n=== Example 6: Quick Functions ===")
    
    # Quick file transcription
    text = transcribe_file("audio.wav", language="vi", model_size="large-v3")
    print(f"Transcribed: {text}")
    
    # Quick microphone transcription
    # text = transcribe_microphone(timeout=10, language="vi")
    # print(f"You said: {text}")


def example_7_different_languages():
    """Example 7: Support for different languages"""
    print("\n=== Example 7: Different Languages ===")
    
    # Vietnamese
    stt_vi = FasterWhisperSTT(model_size="large-v3", language="vi")
    # text = stt_vi.transcribe_from_file("vietnamese_audio.wav")
    
    # English
    stt_en = FasterWhisperSTT(model_size="large-v3", language="en")
    # text = stt_en.transcribe_from_file("english_audio.wav")
    
    # Mandarin
    stt_zh = FasterWhisperSTT(model_size="large-v3", language="zh")
    # text = stt_zh.transcribe_from_file("chinese_audio.wav")
    
    print("✓ Multiple language instances ready")


def example_8_with_error_handling():
    """Example 8: Proper error handling"""
    print("\n=== Example 8: Error Handling ===")
    
    stt = FasterWhisperSTT(model_size="large-v3", language="vi")
    
    # Handle file not found
    try:
        text = stt.transcribe_from_file("nonexistent_audio.wav")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Handle microphone errors
    try:
        text = stt.transcribe_from_microphone(timeout=5)
    except RuntimeError as e:
        print(f"Microphone error: {e}")


def example_9_batch_transcription():
    """Example 9: Batch transcription of multiple files"""
    print("\n=== Example 9: Batch Transcription ===")
    
    import os
    from pathlib import Path
    
    stt = FasterWhisperSTT(model_size="large-v3", language="vi")
    
    audio_dir = Path("path/to/audio/directory")
    results = {}
    
    if audio_dir.exists():
        for audio_file in audio_dir.glob("*.wav"):
            try:
                text = stt.transcribe_from_file(str(audio_file))
                results[audio_file.name] = text
                print(f"✓ {audio_file.name}: {text[:50]}...")
            except Exception as e:
                print(f"✗ {audio_file.name}: {e}")
    
    return results


def example_10_gpu_vs_cpu():
    """Example 10: GPU vs CPU performance comparison"""
    print("\n=== Example 10: Device Selection ===")
    
    # Auto (detects GPU if available)
    stt_auto = FasterWhisperSTT(
        model_size="large-v3",
        device="auto",
        compute_type="default"
    )
    print("STT Engine with auto device detection ready")
    
    # Force CPU
    stt_cpu = FasterWhisperSTT(
        model_size="large-v3",
        device="cpu",
        compute_type="int8"  # Quantized for faster CPU inference
    )
    print("STT Engine with CPU forced ready")
    
    # CUDA GPU (if available)
    # stt_gpu = FasterWhisperSTT(
    #     model_size="large-v3",
    #     device="cuda",
    #     compute_type="float16"
    # )


if __name__ == "__main__":
    print("=" * 60)
    print("Speech-to-Text (STT) Module - Usage Examples")
    print("Using faster-whisper large-v3 model")
    print("=" * 60)
    
    # Uncomment examples to run them
    # example_1_basic_file_transcription()
    # example_2_microphone_input()
    # example_3_with_confidence()
    # example_4_language_detection()
    # example_5_using_global_instance()
    # example_6_quick_convenience_functions()
    example_7_different_languages()
    example_8_with_error_handling()
    # example_9_batch_transcription()
    example_10_gpu_vs_cpu()
    
    print("\n" + "=" * 60)
    print("To use in your code, import from stt module:")
    print("  from stt import FasterWhisperSTT, get_stt_engine")
    print("=" * 60)
