# Convert Image Format

## Overview

A collection of Python scripts for converting images between JPG and PNG with Pillow. It includes a batch converter and individual single-image converters.

**Type:** CLI Utility

## Features

- **Batch conversion** (`convertDynamic.py`): Converts JPG/PNG files below a supplied path to one chosen target format
- **Single JPG to PNG** (`JPGtoPNG.py`): Converts a specified JPG image to a new PNG file
- **Single PNG to JPG** (`PNGtoJPG.py`): Converts a specified PNG image to a new JPEG file
- Refuses to overwrite existing output files

## Dependencies

- `Pillow` — for image format conversion

Install with:

```bash
uv sync
```

## How it works

1. **convertDynamic.py**: Takes an image or directory plus `--to png` or `--to jpeg`, then converts eligible images without overwriting existing counterparts.
2. **JPGtoPNG.py** and **PNGtoJPG.py**: Take explicit source and output paths and use the same conversion logic.

## Project Structure

```
convert_Imgs/
├── convertDynamic.py     # Batch converter (recursive, both directions)
├── JPGtoPNG.py           # Single JPG → PNG converter
├── PNGtoJPG.py           # Single PNG → JPG converter
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Locked dependency versions
├── naruto_first.jpg      # Sample input image (JPG)
├── naruto_first.png      # Sample input image (PNG)
├── naruto_last.jpg       # Sample output image (JPG)
├── naruto_last.png       # Sample output image (PNG)
└── README.md
```

## Setup & Installation

```bash
uv sync
```

## How to Run

**Batch conversion:**
```bash
uv run python convertDynamic.py . --to png
```
This converts JPG images below the current directory to PNG, skipping existing PNG counterparts.

**Single image conversion:**

Provide explicit source and output paths:

```bash
uv run python JPGtoPNG.py naruto_first.jpg naruto.png
uv run python PNGtoJPG.py naruto_first.png naruto.jpg
```

## Testing

No formal test suite present.

## Limitations

- Batch conversion writes alongside each source image.
- Existing target files are skipped rather than overwritten.
- Only JPG, JPEG, and PNG sources are supported.
