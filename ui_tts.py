import time
from pathlib import Path

import gradio as gr

from pipeline_tts import generate_text_stream, sanitize_vietnamese_text
from tts import synthesize_text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def run_pipeline_stream(
    prompt: str,
    model: str,
    system_prompt: str,
    voice: str,
    num_predict: int | None,
    vi_only: bool,
    clean_text: bool,
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Prompt is empty")

    if vi_only:
        system_prompt = (
            system_prompt
            + "\nChi tra loi bang tieng Viet. Khong dung tieng Anh, tieng Trung, emoji, hoac ky tu khong can thiet."
        )

    generated_chunks = []
    for chunk in generate_text_stream(prompt, model, system_prompt, num_predict):
        generated_chunks.append(chunk)
        yield "".join(generated_chunks), None

    generated = "".join(generated_chunks).strip()
    if clean_text:
        generated = sanitize_vietnamese_text(generated)
    if not generated:
        raise gr.Error("LLM returned empty text")

    timestamp = int(time.time())
    text_path = DATA_DIR / f"llm_{timestamp}.txt"
    audio_path = DATA_DIR / f"llm_{timestamp}.wav"

    text_path.write_text(generated, encoding="utf-8")
    synthesize_text(generated, audio_path, voice)

    yield generated, str(audio_path)


with gr.Blocks(title="Vietnamese LLM to TTS") as demo:
    gr.Markdown("# Vietnamese LLM to TTS")
    gr.Markdown("Enter a prompt, generate Vietnamese text with Ollama, then synthesize speech.")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", lines=6, placeholder="Tell me a story in Vietnamese...")

    with gr.Row():
        model = gr.Textbox(label="Ollama model", value="qwen2.5:7b-instruct")
        voice = gr.Textbox(label="Voice", value="ly")

    num_predict = gr.Slider(label="Max tokens", minimum=64, maximum=1024, value=256, step=16)
    vi_only = gr.Checkbox(label="Vietnamese only", value=True)
    clean_text = gr.Checkbox(label="Clean artifacts", value=True)

    system_prompt = gr.Textbox(
        label="System prompt",
        value="You are a helpful assistant that replies in Vietnamese.",
        lines=2,
    )

    run_btn = gr.Button("Generate + Speak")
    output_text = gr.Textbox(label="Generated text", lines=8)
    output_audio = gr.Audio(label="Speech", type="filepath")

    run_btn.click(
        fn=run_pipeline_stream,
        inputs=[prompt, model, system_prompt, voice, num_predict, vi_only, clean_text],
        outputs=[output_text, output_audio],
    )


def main():
    demo.launch()


if __name__ == "__main__":
    main()
