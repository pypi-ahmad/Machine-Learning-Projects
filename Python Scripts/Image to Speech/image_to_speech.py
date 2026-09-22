"""Extract text from an image with Tesseract and create an MP3 with gTTS."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytesseract
from gtts import gTTS
from gtts.tts import gTTSError
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract image text and save it as speech.")
    parser.add_argument("input", type=Path, help="Image to process.")
    parser.add_argument("--language", default="en", help="gTTS language code (default: en).")
    parser.add_argument("--tesseract", type=Path, help="Path to tesseract.exe when it is not on PATH.")
    parser.add_argument("--text-output", type=Path, help="Text output path (default: beside the image).")
    parser.add_argument("--audio-output", type=Path, help="MP3 output path (default: beside the image).")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output files.")
    return parser.parse_args()


def ensure_tesseract(tesseract_path: Path | None) -> None:
    """Configure and validate the local Tesseract executable."""
    if tesseract_path is not None:
        if not tesseract_path.is_file():
            raise ValueError(f"Tesseract executable not found: {tesseract_path}")
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as error:
        raise ValueError(
            "Tesseract is not available. Install it and add it to PATH, or pass --tesseract PATH."
        ) from error


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input image not found: {args.input}")

    text_output = args.text_output or args.input.with_suffix(".txt")
    audio_output = args.audio_output or args.input.with_suffix(".mp3")
    if not args.overwrite and (text_output.exists() or audio_output.exists()):
        raise SystemExit("Output already exists. Choose new paths or pass --overwrite.")

    try:
        ensure_tesseract(args.tesseract)
        with Image.open(args.input) as image:
            text = pytesseract.image_to_string(image)
    except (OSError, ValueError, pytesseract.TesseractError) as error:
        raise SystemExit(f"Could not extract text: {error}") from error

    text = text.strip()
    if not text:
        raise SystemExit("No text was detected in the image.")

    text_output.parent.mkdir(parents=True, exist_ok=True)
    audio_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.write_text(text, encoding="utf-8")
    try:
        gTTS(text=text, lang=args.language, slow=False).save(str(audio_output))
    except (gTTSError, OSError) as error:
        raise SystemExit(f"Could not create speech audio: {error}") from error

    print(f"Saved extracted text to {text_output}")
    print(f"Saved speech audio to {audio_output}")


if __name__ == "__main__":
    main()
