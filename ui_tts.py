import time
import sys
import os
import wave
from pathlib import Path

import gradio as gr
import numpy as np

from pipeline_tts import generate_text_stream, sanitize_vietnamese_text
from tts import synthesize_text, synthesize_text_with_progress, get_available_voices
from speech_input import listen_for_speech, listen_for_wakeword
from stt_vietnamese import get_vietnamese_stt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RECORDED_PROMPT_PATH = DATA_DIR / "recorded_prompt.wav"


VI_STT = get_vietnamese_stt()


def save_wav_from_float32(audio_data: np.ndarray, output_path: Path, sample_rate: int = 16000) -> str:
    """Save float32 mono audio in the range [-1, 1] as a WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio_data, -1.0, 1.0)
    pcm_bytes = (pcm * 32767.0).astype(np.int16).tobytes()

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

    return str(output_path)


def generate_text_only(
    prompt: str,
    model: str,
    system_prompt: str,
    num_predict: int | None,
    vi_only: bool,
    clean_text: bool,
):
    """Generate text ONLY - no synthesis. Stream text as it's generated."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Prompt is empty")

    if vi_only:
        system_prompt = (
            system_prompt
            + "\nChi tra loi bang tieng Viet. Khong dung tieng Anh, tieng Trung, emoji, hoac ky tu khong can thiet."
        )

    # Stream text generation
    generated_chunks = []
    for chunk in generate_text_stream(prompt, model, system_prompt, num_predict):
        generated_chunks.append(chunk)
        partial = "".join(generated_chunks)
        yield partial

    # Final cleaned text
    generated = "".join(generated_chunks).strip()
    if clean_text:
        generated = sanitize_vietnamese_text(generated)
    if not generated:
        raise gr.Error("LLM returned empty text")

    timestamp = int(time.time())
    text_path = DATA_DIR / f"llm_{timestamp}.txt"
    text_path.write_text(generated, encoding="utf-8")
    
    yield generated


def synthesize_speech_only(
    generated_text: str,
    voice: str,
):
    """Synthesize speech with progress shown in console."""
    if not generated_text or not generated_text.strip():
        raise gr.Error("No text to synthesize. Generate text first!")
    
    # Find the most recent text file to get timestamp
    import glob
    text_files = sorted(glob.glob(str(DATA_DIR / "llm_*.txt")))
    if not text_files:
        raise gr.Error("Text file not found")
    
    text_file = text_files[-1]
    timestamp = text_file.split("_")[-1].replace(".txt", "")
    audio_path = DATA_DIR / f"llm_{timestamp}.wav"
    
    synthesize_text(generated_text, audio_path, voice)
    
    return str(audio_path)


def synthesize_speech_with_progress_ui(
    generated_text: str,
    voice: str,
):
    """Synthesize speech with real-time progress updates shown in UI and console."""
    if not generated_text or not generated_text.strip():
        raise gr.Error("No text to synthesize. Generate text first!")
    
    # Find the most recent text file to get timestamp
    import glob
    text_files = sorted(glob.glob(str(DATA_DIR / "llm_*.txt")))
    if not text_files:
        raise gr.Error("Text file not found")
    
    text_file = text_files[-1]
    timestamp = text_file.split("_")[-1].replace(".txt", "")
    audio_path = DATA_DIR / f"llm_{timestamp}.wav"
    
    # Stream progress updates in real-time
    for progress_tuple in synthesize_text_with_progress(generated_text, audio_path, voice):
        current, total = progress_tuple
        progress_msg = f"Synthesizing: [{current}/{total}]"
        # Yield progress and None for audio (will be filled at end)
        yield progress_msg, None
    
    # Final yield with completed audio file
    yield "✓ Synthesis complete!", str(audio_path)


def run_pipeline_stream(
    prompt: str,
    model: str,
    system_prompt: str,
    voice: str,
    num_predict: int | None,
    vi_only: bool,
    clean_text: bool,
):
    """Combined pipeline - text then speech."""
    # First: Generate text with streaming
    final_text = None
    for text_update in generate_text_only(prompt, model, system_prompt, num_predict, vi_only, clean_text):
        final_text = text_update
        yield text_update, None
    
    # Second: Synthesize speech
    audio_path = synthesize_speech_only(final_text, voice)
    
    # Final: Yield complete result
    yield final_text, audio_path


def listen_speech_input():
    """Record Vietnamese speech, save a preview WAV, and return the file path."""
    try:
        audio_data = VI_STT.record_microphone_audio(timeout=10, phrase_time_limit=10)
        return save_wav_from_float32(audio_data, RECORDED_PROMPT_PATH)
    except Exception as e:
        raise gr.Error(f"Speech recognition error: {str(e)}")


def transcribe_recorded_prompt(audio_path):
    """Transcribe the recorded preview WAV into the prompt box."""
    if not audio_path:
        raise gr.Error("Record audio first, then transcribe it.")

    try:
        text = VI_STT.transcribe_file(str(audio_path))
        if not text:
            raise gr.Error("Could not transcribe the recorded audio. Please try again.")
        return text
    except Exception as e:
        raise gr.Error(f"Vietnamese STT error: {str(e)}")


def listen_wakeword_input():
    """Listen for wakeword and return the prompt after it."""
    try:
        text = listen_for_wakeword(wakeword="hey pi-bot", timeout=30)
        if not text:
            raise gr.Error('Wakeword "hey pi-bot" not detected. Please try again.')
        return text
    except Exception as e:
        raise gr.Error(f"Wakeword detection error: {str(e)}")


with gr.Blocks(title="Vietnamese LLM to TTS") as demo:
    gr.Markdown("# Vietnamese LLM to TTS")
    gr.Markdown("Step 1: Record and preview a Vietnamese prompt, then transcribe it into the prompt box. Step 2: Generate text. Step 3: Synthesize speech.")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", lines=6, placeholder="Tell me a story in Vietnamese...")

    with gr.Row():
        btn_listen = gr.Button("🎤 Record Prompt", variant="primary")
        btn_transcribe_recording = gr.Button("📝 Transcribe Recording", variant="secondary")
        btn_wakeword = gr.Button("🎤 Listen for 'hey pi-bot'", variant="primary")

    recorded_audio = gr.Audio(label="Recorded Prompt Preview", type="filepath")
    recorded_status = gr.Textbox(label="Recording Status", interactive=False, lines=2)

    with gr.Row():
        model = gr.Textbox(label="Ollama model", value="gemma3-tts")
        voice = gr.Dropdown(label="Voice", choices=get_available_voices(), value="Ly")

    num_predict = gr.Slider(label="Max tokens", minimum=64, maximum=1024, value=256, step=16)
    vi_only = gr.Checkbox(label="Vietnamese only", value=True)
    clean_text = gr.Checkbox(label="Clean artifacts", value=True)

    system_prompt = gr.Textbox(
        label="System prompt",
        value="You are a helpful assistant that replies in Vietnamese.",
        lines=2,
    )

    with gr.Row():
        btn_generate = gr.Button("📝 Generate Text", variant="primary")
        btn_synthesize = gr.Button("🔊 Synthesize Speech", variant="primary")

    output_text = gr.Textbox(label="Generated text", lines=8)
    output_audio = gr.Audio(label="Speech", type="filepath")
    output_progress = gr.Textbox(label="Progress", interactive=False, lines=3)

    # Speech input handlers
    btn_listen.click(fn=listen_speech_input, outputs=recorded_audio)
    btn_listen.click(
        fn=lambda: "Recorded audio is ready to preview. Click 'Transcribe Recording' to fill the prompt.",
        outputs=recorded_status,
    )
    btn_transcribe_recording.click(fn=transcribe_recorded_prompt, inputs=recorded_audio, outputs=prompt)
    btn_wakeword.click(fn=listen_wakeword_input, outputs=prompt)

    # Step 1: Generate text ONLY - streams to text field, NO progress bar in audio field
    btn_generate.click(
        fn=generate_text_only,
        inputs=[prompt, model, system_prompt, num_predict, vi_only, clean_text],
        outputs=output_text,
    )

    # Step 2: Synthesize speech with progress - shows progress in UI and console
    btn_synthesize.click(
        fn=synthesize_speech_with_progress_ui,
        inputs=[output_text, voice],
        outputs=[output_progress, output_audio],
    )


def main():
    demo.launch()


if __name__ == "__main__":
    main()
