"""Extract tables from one PDF into a CSV file with tabula-py."""

import argparse
from pathlib import Path

import tabula
from tabula.errors import JavaNotFoundError


def convert_pdf(input_path: Path, output_path: Path, pages: str) -> None:
    """Write tables from the selected PDF to one CSV file."""
    tabula.convert_into(
        input_path,
        output_path,
        output_format="csv",
        pages=pages,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF tables into CSV.")
    parser.add_argument("input", type=Path, help="PDF file containing tables")
    parser.add_argument("--output", type=Path, help="Destination CSV file")
    parser.add_argument("--pages", default="all", help="Page range accepted by tabula-py (default: all)")
    args = parser.parse_args()

    if not args.input.is_file() or args.input.suffix.lower() != ".pdf":
        parser.error("input must be an existing PDF file")

    output_path = args.output or args.input.with_suffix(".csv")
    try:
        convert_pdf(args.input, output_path, args.pages)
    except JavaNotFoundError:
        parser.exit(1, "Java was not found. Install a JRE or JDK and add it to PATH.\n")
    except OSError as error:
        parser.exit(1, f"Unable to convert PDF: {error}\n")

    print(f"Extracted tables from {args.input} to {output_path}")


if __name__ == "__main__":
    main()
