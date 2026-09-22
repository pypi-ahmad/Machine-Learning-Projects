"""Convert a JSON array of objects into CSV."""

import argparse
import csv
import json
from pathlib import Path


def convert(input_path: Path, output_path: Path) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or not all(isinstance(row, dict) for row in data):
        raise ValueError("Input JSON must be a non-empty array of objects.")
    columns = list(dict.fromkeys(key for row in data for key in row))
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a JSON array of objects to CSV.")
    parser.add_argument("input", type=Path, help="Source JSON file.")
    parser.add_argument("output", type=Path, help="Destination CSV file.")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input and output paths must differ.")
    try:
        convert(args.input, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Conversion failed: {error}") from error
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
