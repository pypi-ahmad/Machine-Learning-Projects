"""OCR one selected PDF into a local UTF-8 text file."""

import argparse
import io
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image


def ocr_page(page: pymupdf.Page, dpi: int) -> str:
    """Render one PDF page locally and extract text with Tesseract."""
    pixels = page.get_pixmap(dpi=dpi)
    with Image.open(io.BytesIO(pixels.tobytes("png"))) as image:
        return pytesseract.image_to_string(image)


def extract_pdf(path: Path, dpi: int) -> str:
    """OCR every page in a PDF without creating intermediate image files."""
    with pymupdf.open(path) as document:
        return "\n".join(map(lambda page: ocr_page(page, dpi), document)).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR a PDF with local Tesseract.")
    parser.add_argument("input", type=Path, help="PDF file to OCR")
    parser.add_argument("--output", type=Path, help="Destination text file")
    parser.add_argument("--dpi", type=int, default=300, help="Render resolution (default: 300)")
    args = parser.parse_args()

    if not args.input.is_file() or args.input.suffix.lower() != ".pdf":
        parser.error("input must be an existing PDF file")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")

    try:
        text = extract_pdf(args.input, args.dpi)
    except pytesseract.TesseractNotFoundError:
        parser.exit(1, "Tesseract was not found. Install it and add it to PATH.\n")
    except pymupdf.FileDataError as error:
        parser.exit(1, f"Unable to read PDF: {error}\n")

    if not text:
        parser.exit(1, "No text was extracted from the PDF.\n")
    output = args.output or args.input.with_suffix(".txt")
    output.write_text(text, encoding="utf-8")
    print(f"Extracted text from {args.input} to {output}")


if __name__ == "__main__":
    main()
