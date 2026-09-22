"""Convert UTF-8 text from a file into an MP3 with Google Text-to-Speech."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

from gtts import gTTS
from gtts.tts import gTTSError


def save_speech(
    text: str, output: Path, language: str, slow: bool, factory: Callable[..., gTTS] = gTTS
) -> None:
    """Generate one new MP3 without replacing an existing output file."""
    if not text.strip():
        raise ValueError("Input text file is empty.")
    if output.suffix.lower() != ".mp3":
        raise ValueError("Output file must use the .mp3 extension.")
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose another output path")
    factory(text=text, lang=language, slow=slow).save(str(output))


def main() -> None:
    """Read a text file, create an MP3, and optionally open it on Windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 text input file")
    parser.add_argument("--output", type=Path, default=Path("voice.mp3"))
    parser.add_argument("--language", default="en", help="gTTS language code")
    parser.add_argument("--slow", action="store_true", help="generate slower speech")
    parser.add_argument("--play", action="store_true", help="open the generated MP3 with Windows' default player")
    args = parser.parse_args()
    try:
        save_speech(args.input.read_text(encoding="utf-8"), args.output, args.language, args.slow)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, gTTSError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Saved speech to {args.output}")
    if args.play:
        os.startfile(args.output)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
