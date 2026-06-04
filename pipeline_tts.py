import argparse
import json
import sys
import unicodedata
from pathlib import Path

import requests

from tts import synthesize_text

BASE_DIR = Path(__file__).resolve().parent


def resolve_output_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def get_prompt(args) -> str:
    if args.prompt:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return input("Enter prompt: ").strip()


def generate_text_stream(
    prompt: str,
    model: str,
    system_prompt: str,
    num_predict: int | None = None,
    timeout: int = 600,
):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
    }
    if num_predict is not None:
        payload["options"] = {"num_predict": num_predict}

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        stream=True,
        timeout=(5, timeout),
    )
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        chunk = data.get("response") or ""
        if chunk:
            yield chunk
        if data.get("done"):
            break


def generate_text(
    prompt: str,
    model: str,
    system_prompt: str,
    num_predict: int | None = None,
    timeout: int = 600,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
    }
    if num_predict is not None:
        payload["options"] = {"num_predict": num_predict}

    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return (data.get("response") or "").strip()


def sanitize_vietnamese_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    cleaned_chars = []
    for ch in text:
        if ch.isdigit():
            continue
        if ch.isalpha() or ch.isspace() or ch in {".", ",", "!", "?", ":", ";", "-", "\n"}:
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate Vietnamese text via Ollama and synthesize with Vieneu TTS")
    parser.add_argument("--prompt", help="Prompt text for the LLM (if omitted, reads stdin or asks interactively)")
    parser.add_argument("--model", default="gemma3-tts", help="Ollama model name")
    parser.add_argument("--system", default="You are a helpful assistant that replies in Vietnamese.", help="System prompt")
    parser.add_argument("--vi-only", action="store_true", help="Force Vietnamese-only responses")
    parser.add_argument("--clean-text", action="store_true", help="Clean non-Vietnamese artifacts from output")
    parser.add_argument("--voice", default="ly", help="Voice name or prefix for TTS")
    parser.add_argument("--output", default="data/llm.wav", help="Output WAV file")
    parser.add_argument("--save-text", default="data/llm.txt", help="Save generated text to this file (set empty to skip)")
    parser.add_argument("--num-predict", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--timeout", type=int, default=600, help="Ollama read timeout in seconds")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    prompt = get_prompt(args)
    if not prompt:
        print("Error: prompt is empty")
        sys.exit(1)

    system_prompt = args.system
    if args.vi_only:
        system_prompt = (
            system_prompt
            + "\nChi tra loi bang tieng Viet. Khong dung tieng Anh, tieng Trung, emoji, hoac ky tu khong can thiet."
        )

    print("Generating text with Ollama...")
    if args.no_stream:
        generated = generate_text(prompt, args.model, system_prompt, args.num_predict, args.timeout)
    else:
        generated_chunks = []
        try:
            for chunk in generate_text_stream(prompt, args.model, system_prompt, args.num_predict, args.timeout):
                print(chunk, end="", flush=True)
                generated_chunks.append(chunk)
            print()
        except requests.exceptions.ReadTimeout:
            print("\nError: Ollama request timed out while streaming")
            sys.exit(1)

        generated = "".join(generated_chunks).strip()
    if args.clean_text:
        generated = sanitize_vietnamese_text(generated)

    if not generated:
        print("Error: LLM returned empty text")
        sys.exit(1)

    if args.save_text:
        save_path = resolve_output_path(args.save_text)
        save_path.write_text(generated, encoding="utf-8")
        print(f"Saved text: {save_path}")

    output_path = resolve_output_path(args.output)
    synthesize_text(generated, output_path, args.voice)


if __name__ == "__main__":
    main()
