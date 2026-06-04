import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from vieneu import Vieneu

# Run everything relative to this file so the folder can work standalone.
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def get_available_voices():
    """Get list of available voice IDs from vieneu model."""
    try:
        tts = Vieneu()
        voices = tts.list_preset_voices()
        voice_ids = [vid for desc, vid in voices]
        print(f"✓ Loaded voices: {voice_ids}")
        return voice_ids
    except Exception as e:
        print(f"✗ Error loading voices: {e}")
        return ["ly", "binh", "tuyen", "vinh", "doan", "son", "ngoc"]


def select_voice(tts: Vieneu, voice_query: str):
    voices = tts.list_preset_voices()
    print("\nDanh sách giọng:")
    for desc, vid in voices:
        print(f"  {desc} --> ID: '{vid}'")

    target_voice_id = None
    for desc, vid in voices:
        if voice_query.lower() in desc.lower() or voice_query.lower() in vid.lower():
            target_voice_id = vid
            print(f"\nDùng giọng: {desc} (ID: {vid})")
            break

    if not target_voice_id:
        target_voice_id = voices[0][1]
        print(f"\nKhông tìm thấy giọng '{voice_query}', dùng mặc định: {voices[0][0]}")

    return tts.get_preset_voice(target_voice_id)


def split_sentences(text_full: str):
    sentences = re.split(r'(?<=[.!?])\s+', text_full)
    return [s.strip() for s in sentences if s.strip()]


def synthesize_text(text_full: str, output_path: Path, voice_query: str = "ly"):
    tts = Vieneu()
    voice_data = select_voice(tts, voice_query)

    sentences = split_sentences(text_full)
    print(f"\nTổng số câu: {len(sentences)}")

    all_audio = []
    silence = None

    for i, sentence in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] {sentence[:60]}...")
        chunk = tts.infer(text=sentence, voice=voice_data)

        if hasattr(chunk, "audio"):
            arr = chunk.audio
        elif isinstance(chunk, np.ndarray):
            arr = chunk
        else:
            arr = np.array(chunk)

        if silence is None:
            silence = np.zeros(int(24000 * 0.3), dtype=arr.dtype)

        all_audio.append(arr)
        all_audio.append(silence)

    final_audio = np.concatenate(all_audio)
    sf.write(output_path, final_audio, samplerate=24000)
    print(f"\nHoàn thành! Đã lưu: {output_path}")
    print(f"Thời lượng: {len(final_audio)/24000:.1f} giây")


def synthesize_text_with_progress(text_full: str, output_path: Path, voice_query: str = "ly", progress_callback=None):
    """Synthesize text with progress updates for UI. Yields progress as it processes."""
    tts = Vieneu()
    voice_data = select_voice(tts, voice_query)

    sentences = split_sentences(text_full)
    total = len(sentences)
    print(f"\nTổng số câu: {total}")

    all_audio = []
    silence = None

    for i, sentence in enumerate(sentences):
        current = i + 1
        progress_msg = f"  [{current}/{total}] {sentence[:60]}..."
        print(progress_msg)
        
        # Yield progress for UI
        yield (current, total)
        
        # Also call callback if provided (for backward compatibility)
        if progress_callback:
            progress_callback((current, total))
        
        chunk = tts.infer(text=sentence, voice=voice_data)

        if hasattr(chunk, "audio"):
            arr = chunk.audio
        elif isinstance(chunk, np.ndarray):
            arr = chunk
        else:
            arr = np.array(chunk)

        if silence is None:
            silence = np.zeros(int(24000 * 0.3), dtype=arr.dtype)

        all_audio.append(arr)
        all_audio.append(silence)

    final_audio = np.concatenate(all_audio)
    sf.write(output_path, final_audio, samplerate=24000)
    print(f"\nHoàn thành! Đã lưu: {output_path}")
    print(f"Thời lượng: {len(final_audio)/24000:.1f} giây")


def main():
    parser = argparse.ArgumentParser(description="Chuyển văn bản tiếng Việt thành giọng nói")
    parser.add_argument("--input", default="data/truyen.txt", help="File văn bản đầu vào (mặc định: data/truyen.txt)")
    parser.add_argument("--output", default="data/truyen.wav", help="File âm thanh đầu ra  (mặc định: data/truyen.wav)")
    parser.add_argument("--voice", default="ly", help="Tên giọng đọc         (mặc định: ly)")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text_full = f.read().strip()
        print(f"Đã đọc: {input_path} ({len(text_full)} ký tự)")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{input_path}'")
        sys.exit(1)

    synthesize_text(text_full, output_path, args.voice)


if __name__ == "__main__":
    main()