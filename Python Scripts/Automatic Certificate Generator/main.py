"""Generate PNG certificates from a CSV name list and image template."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_names(path: Path) -> list[str]:
    """Read non-empty names from a CSV file with a ``name`` column."""
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or "name" not in reader.fieldnames:
            raise ValueError("CSV requires a 'name' column.")
        return [row["name"].strip() for row in reader if row["name"].strip()]


def safe_filename(name: str, index: int) -> str:
    """Create a portable unique PNG filename."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "certificate"
    return f"{stem}-{index}.png"


def generate(names: list[str], template: Path, output_dir: Path, font_path: Path | None, font_size: int, position: tuple[int, int]) -> list[Path]:
    """Render each name on a fresh template image."""
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, name in enumerate(names, 1):
        image = Image.open(template).convert("RGB")
        ImageDraw.Draw(image).text(position, name, fill="black", font=font)
        output = output_dir / safe_filename(name, index)
        image.save(output)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("pictures"))
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-size", type=int, default=60)
    parser.add_argument("--x", type=int, default=150)
    parser.add_argument("--y", type=int, default=250)
    args = parser.parse_args()
    try:
        outputs = generate(load_names(args.csv), args.template, args.output_dir, args.font, args.font_size, (args.x, args.y))
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Generated {len(outputs)} certificate(s) in {args.output_dir}.")


if __name__ == "__main__":
    main()
