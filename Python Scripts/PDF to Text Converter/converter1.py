"""Extract embedded text from one PDF into a UTF-8 text file."""

import argparse
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def extract_pdf_text(path: Path) -> str:
    """Return text embedded in each page of a PDF."""
    reader = PdfReader(path)
    return "\n".join(map(lambda page: page.extract_text() or "", reader.pages)).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embedded PDF text into a UTF-8 file.")
    parser.add_argument("input", type=Path, help="PDF file to read")
    parser.add_argument("--output", type=Path, help="Destination text file")
    args = parser.parse_args()

    if not args.input.is_file() or args.input.suffix.lower() != ".pdf":
        parser.error("input must be an existing PDF file")

    try:
        text = extract_pdf_text(args.input)
    except PdfReadError as error:
        parser.exit(1, f"Unable to read PDF: {error}\n")

    if not text:
        parser.exit(1, "No embedded text was found. This may be a scanned PDF that needs OCR.\n")

    output = args.output or args.input.with_suffix(".txt")
    output.write_text(text, encoding="utf-8")
    print(f"Extracted text from {args.input} to {output}")


if __name__ == "__main__":
    main()
