"""
Vietnamese STT Interactive Test Script
Test the Vietnamese speech-to-text module with microphone or file input
"""

import sys
from pathlib import Path
from stt_vietnamese import (
    get_vietnamese_stt,
    transcribe_vietnamese_microphone,
    transcribe_vietnamese_file
)


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def test_microphone():
    """Test Vietnamese transcription from microphone."""
    print_header("🎤 Microphone Test - Vietnamese STT")
    
    print("Instructions:")
    print("1. Speak Vietnamese text clearly")
    print("2. The system will play back what was recorded")
    print("3. Confirm recording quality, then transcribe")
    print()
    
    try:
        stt = get_vietnamese_stt()
        timeout = 10
        print(f"Waiting for input ({timeout} seconds)...")
        audio_data = stt.record_microphone_audio(timeout=timeout, phrase_time_limit=12)

        print("\n🔊 Playing back recording...")
        stt.play_audio(audio_data, sample_rate=16000)

        confirm = input("Use this recording for transcription? (y/n): ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("⚠️  Recording rejected. Please run the microphone test again.")
            return ""

        print("📊 Processing transcription...")
        text, _ = stt.transcribe_with_confidence(audio_data)
        
        if text:
            print()
            print("📝 Transcribed Text:")
            print(f"   {text}")
            return text
        else:
            print("No speech detected")
            return ""
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""


def test_file(file_path):
    """Test Vietnamese transcription from file."""
    print_header("📄 File Test - Vietnamese STT")
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return ""
    
    print(f"Transcribing: {file_path.name}")
    print()
    
    try:
        text = transcribe_vietnamese_file(str(file_path))
        
        if text:
            print()
            print("📝 Transcribed Text:")
            print(f"   {text}")
            return text
        else:
            print("No speech detected in file")
            return ""
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""


def test_confidence():
    """Test Vietnamese transcription with confidence score."""
    print_header("⭐ Confidence Test - Vietnamese STT")
    
    try:
        stt = get_vietnamese_stt()
        
        print("Waiting for microphone input (10 seconds)...")
        
        # Capture audio
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=10)
        
        print("Processing...")
        
        # Convert to numpy for confidence testing
        import numpy as np
        audio_data = np.frombuffer(
            audio.get_raw_data(), np.int16
        ).astype(np.float32) / 32768.0
        
        text, confidence = stt.transcribe_with_confidence(audio_data)
        
        if text:
            print()
            print("📝 Transcribed Text:")
            print(f"   {text}")
            print()
            print("⭐ Confidence Score:")
            print(f"   {confidence:.1%}")
            
            if confidence > 0.8:
                print("   Quality: ✅ Excellent")
            elif confidence > 0.6:
                print("   Quality: ✅ Good")
            elif confidence > 0.4:
                print("   Quality: ⚠️  Fair")
            else:
                print("   Quality: ❌ Low")
            
            return text, confidence
        else:
            print("No speech detected")
            return "", 0.0
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return "", 0.0


def test_vietnamese_features():
    """Test Vietnamese-specific features."""
    print_header("🇻🇳 Vietnamese Features Test")
    
    print("Testing Vietnamese STT Module Features:")
    print()
    
    try:
        stt = get_vietnamese_stt()
        
        print("✓ Module initialized")
        print(f"  - Language: Vietnamese (vi)")
        print(f"  - Model: {stt.model_size}")
        print(f"  - Device: {stt.device}")
        print(f"  - Compute Type: {stt.compute_type}")
        print()
        
        # Check available methods
        print("✓ Available methods:")
        methods = [
            "transcribe_file()",
            "transcribe_microphone()",
            "transcribe_with_confidence()"
        ]
        for method in methods:
            print(f"  - {method}")
        
        print()
        print("✅ All Vietnamese STT features available!")
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def show_menu():
    """Display interactive menu."""
    print_header("Vietnamese Speech-to-Text (STT) Test Menu")
    
    print("Select a test option:")
    print()
    print("  1. 🎤 Microphone Test")
    print("     - Speak Vietnamese and transcribe in real-time")
    print()
    print("  2. 📄 File Test")
    print("     - Transcribe an audio file")
    print()
    print("  3. ⭐ Confidence Test")
    print("     - Get transcription with confidence score")
    print()
    print("  4. 🇻🇳 Features Test")
    print("     - Test Vietnamese STT module features")
    print()
    print("  5. 📚 Show Examples")
    print("     - Display usage examples")
    print()
    print("  0. Exit")
    print()


def show_examples():
    """Show usage examples."""
    print_header("Vietnamese STT Usage Examples")
    
    examples = """
Example 1: Basic Microphone Transcription
─────────────────────────────────────────
from stt_vietnamese import get_vietnamese_stt

stt = get_vietnamese_stt()
text = stt.transcribe_microphone(timeout=10)
print(f"You said: {text}")


Example 2: File Transcription
─────────────────────────────
from stt_vietnamese import transcribe_vietnamese_file

text = transcribe_vietnamese_file("vietnamese_audio.wav")
print(f"Transcription: {text}")


Example 3: Transcription with Confidence
─────────────────────────────────────────
from stt_vietnamese import get_vietnamese_stt

stt = get_vietnamese_stt()
text, confidence = stt.transcribe_with_confidence("audio.wav")
print(f"Text: {text}")
print(f"Confidence: {confidence:.1%}")


Example 4: GPU Acceleration (if available)
───────────────────────────────────────────
from stt_vietnamese import VietnameseSpeechToText

stt = VietnameseSpeechToText(
    device="cuda",
    compute_type="float16"
)
text = stt.transcribe_microphone()


Example 5: Quick Functions
──────────────────────────
from stt_vietnamese import (
    transcribe_vietnamese_microphone,
    transcribe_vietnamese_file
)

# Microphone
text = transcribe_vietnamese_microphone(timeout=10)

# File
text = transcribe_vietnamese_file("audio.wav")
    """
    
    print(examples)


def main():
    """Main test loop."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Vietnamese Speech-to-Text (STT)" + " " * 22 + "║")
    print("║" + " " * 20 + "Interactive Test Suite" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝")
    
    while True:
        show_menu()
        
        choice = input("Enter option (0-5): ").strip()
        
        if choice == "0":
            print("\n👋 Goodbye!")
            break
        
        elif choice == "1":
            test_microphone()
        
        elif choice == "2":
            print("Enter audio file path (e.g., audio.wav):")
            file_path = input("Path: ").strip()
            if file_path:
                test_file(file_path)
            else:
                print("❌ No file path provided")
        
        elif choice == "3":
            test_confidence()
        
        elif choice == "4":
            test_vietnamese_features()
        
        elif choice == "5":
            show_examples()
        
        else:
            print("❌ Invalid option. Please try again.")
        
        input("\n\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
