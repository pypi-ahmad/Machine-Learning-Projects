"""Extract text from a PDF and create a local audiobook file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyttsx3
from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract readable text from every page of a PDF."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
    if not text.strip():
        raise ValueError("No extractable text was found. The PDF may be scanned.")
    return text


def create_audio(text: str, output_path: Path, speak: bool) -> None:
    """Save text through the local TTS engine and optionally speak it."""
    engine = pyttsx3.init()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save_to_file(text, str(output_path))
    if speak:
        engine.say(text)
    engine.runAndWait()
    engine.stop()


def main() -> None:
    """Parse CLI arguments and create an audiobook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="input PDF path")
    parser.add_argument("--output", type=Path, default=Path("audiobook.wav"))
    parser.add_argument("--speak", action="store_true", help="also play the text aloud")
    args = parser.parse_args()
    try:
        text = extract_pdf_text(args.pdf)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    create_audio(text, args.output, args.speak)
    print(f"Created audiobook from {args.pdf} at {args.output}")


if __name__ == "__main__":
    main()
