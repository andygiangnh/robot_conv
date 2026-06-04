import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from vieneu import Vieneu

# Run everything relative to this file so the folder can work standalone.
BASE_DIR = Path(__file__).resolve().parent

# ── Tham số dòng lệnh ────────────────────────────────────────
parser = argparse.ArgumentParser(description="Chuyển văn bản tiếng Việt thành giọng nói")
parser.add_argument("--input",  default="data/truyen.txt", help="File văn bản đầu vào (mặc định: data/truyen.txt)")
parser.add_argument("--output", default="data/truyen.wav",  help="File âm thanh đầu ra  (mặc định: data/truyen.wav)")
parser.add_argument("--voice",  default="ly",              help="Tên giọng đọc         (mặc định: ly)")
args = parser.parse_args()

input_path = Path(args.input)
if not input_path.is_absolute():
    input_path = BASE_DIR / input_path

output_path = Path(args.output)
if not output_path.is_absolute():
    output_path = BASE_DIR / output_path

# ── 1. Đọc file đầu vào ──────────────────────────────────────
try:
    with open(input_path, "r", encoding="utf-8") as f:
        text_full = f.read().strip()
    print(f"Đã đọc: {input_path} ({len(text_full)} ký tự)")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{input_path}'")
    sys.exit(1)

# ── 2. Khởi tạo TTS ──────────────────────────────────────────
tts = Vieneu()

# ── 3. Chọn giọng đọc ────────────────────────────────────────
voices = tts.list_preset_voices()
print("\nDanh sách giọng:")
for desc, vid in voices:
    print(f"  {desc} --> ID: '{vid}'")

target_voice_id = None
for desc, vid in voices:
    if args.voice.lower() in desc.lower() or args.voice.lower() in vid.lower():
        target_voice_id = vid
        print(f"\nDùng giọng: {desc} (ID: {vid})")
        break

if not target_voice_id:
    target_voice_id = voices[0][1]
    print(f"\nKhông tìm thấy giọng '{args.voice}', dùng mặc định: {voices[0][0]}")

voice_data = tts.get_preset_voice(target_voice_id)

# ── 4. Tách câu ──────────────────────────────────────────────
sentences = re.split(r'(?<=[.!?])\s+', text_full)
sentences = [s.strip() for s in sentences if s.strip()]
print(f"\nTổng số câu: {len(sentences)}")

# ── 5. Tổng hợp từng câu ─────────────────────────────────────
all_audio = []
silence = None

for i, sentence in enumerate(sentences):
    print(f"  [{i+1}/{len(sentences)}] {sentence[:60]}...")
    chunk = tts.infer(text=sentence, voice=voice_data)

    if hasattr(chunk, 'audio'):
        arr = chunk.audio
    elif isinstance(chunk, np.ndarray):
        arr = chunk
    else:
        arr = np.array(chunk)

    if silence is None:
        silence = np.zeros(int(24000 * 0.3), dtype=arr.dtype)

    all_audio.append(arr)
    all_audio.append(silence)

# ── 6. Ghép và lưu file ──────────────────────────────────────
final_audio = np.concatenate(all_audio)
sf.write(output_path, final_audio, samplerate=24000)
print(f"\nHoàn thành! Đã lưu: {output_path}")
print(f"Thời lượng: {len(final_audio)/24000:.1f} giây")