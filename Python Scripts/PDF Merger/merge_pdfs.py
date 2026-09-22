"""Merge PDF files in order, with an optional PDF inserted at a page index."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfWriter


def merge_pdfs(inputs: list[Path], output: Path, inserted: Path | None, insert_at: int) -> None:
    """Append input PDFs and optionally insert one PDF at a page position."""
    resolved_output = output.resolve()
    resolved_inputs = set(map(Path.resolve, inputs))
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose another output path")
    if resolved_output in resolved_inputs or (
        inserted is not None and resolved_output == inserted.resolve()
    ):
        raise ValueError("Output path must not be one of the input PDFs")
    if not all(map(Path.is_file, inputs)):
        raise FileNotFoundError("Every input PDF must exist")
    if inserted is not None and not inserted.is_file():
        raise FileNotFoundError(f"Inserted PDF does not exist: {inserted}")

    writer = PdfWriter()
    try:
        for input_pdf in inputs:
            writer.append(input_pdf)
        if inserted is not None:
            writer.merge(insert_at, inserted)
        with output.open("xb") as destination:
            writer.write(destination)
    finally:
        writer.close()


def main() -> None:
    """Parse PDF paths and create one merged output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="PDFs to append in order")
    parser.add_argument("--output", required=True, type=Path, help="new merged PDF path")
    parser.add_argument("--insert", type=Path, help="extra PDF to insert")
    parser.add_argument("--insert-at", type=int, default=0, help="zero-based output page index")
    args = parser.parse_args()
    if args.insert_at < 0:
        parser.error("--insert-at must be zero or greater")
    try:
        merge_pdfs(args.inputs, args.output, args.insert, args.insert_at)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Merged {len(args.inputs)} PDF(s) into {args.output}")


if __name__ == "__main__":
    main()
