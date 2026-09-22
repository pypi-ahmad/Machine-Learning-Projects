"""Extract text from a PDF or image, save it locally, and optionally read it aloud."""

import argparse
import io
from pathlib import Path

import pymupdf
import pyttsx3
import pytesseract
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def extract_image_text(path: Path) -> str:
    """Extract text from one image with the local Tesseract executable."""
    with Image.open(path) as image:
        return pytesseract.image_to_string(image)


def ocr_page(page: pymupdf.Page) -> str:
    """Render a PDF page locally and send its pixels to Tesseract."""
    pixels = page.get_pixmap(dpi=300)
    with Image.open(io.BytesIO(pixels.tobytes("png"))) as image:
        return pytesseract.image_to_string(image)


def extract_pdf_text(path: Path, use_ocr: bool) -> str:
    """Extract embedded PDF text, with optional OCR for scanned pages."""
    with pymupdf.open(path) as document:
        text = "\n".join(map(lambda page: page.get_text(), document)).strip()
        if text or not use_ocr:
            return text
        return "\n".join(map(ocr_page, document)).strip()


def speak(text: str) -> None:
    """Read text using the configured local text-to-speech engine."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read text from a PDF or image and optionally speak it.")
    parser.add_argument("input", type=Path, help="PDF or image to process")
    parser.add_argument("--ocr", action="store_true", help="OCR a PDF when it has no embedded text")
    parser.add_argument("--no-speak", action="store_true", help="Do not use local text-to-speech")
    parser.add_argument("--output", type=Path, help="Destination text file")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")

    suffix = args.input.suffix.lower()
    if suffix == ".pdf":
        text = extract_pdf_text(args.input, args.ocr)
    elif suffix in IMAGE_SUFFIXES:
        text = extract_image_text(args.input)
    else:
        parser.error("input must be a PDF, BMP, JPEG, PNG, or TIFF image")

    if not text:
        parser.exit(1, "No text was extracted. Use --ocr for scanned PDFs.\n")

    output = args.output or args.input.with_suffix(".txt")
    output.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved text to {output}")
    if not args.no_speak:
        speak(text)


if __name__ == "__main__":
    main()
