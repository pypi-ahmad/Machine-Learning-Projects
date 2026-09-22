# Convert Image to PDF

## Overview

A command-line utility that converts JPG, JPEG, or PNG images to PDF format using `img2pdf`. It supports a single image or all supported images in a directory.

**Type:** CLI Utility

## Features

- Convert a single JPG, JPEG, or PNG image to a PDF
- Convert supported images in a directory into a single multi-page PDF
- Automatic detection of whether the input is a file or directory
- Lossless conversion (img2pdf does not re-encode the image)
- Deterministic directory ordering and overwrite protection

## Dependencies

Managed in `pyproject.toml` and locked in `uv.lock`:

| Package | Version |
|---------|---------|
| img2pdf | Managed by uv |

Additional standard library imports: `argparse`, `os`, `pathlib`

## How It Works

1. The script reads a file or directory path from the CLI.
2. **If the path is a directory:**
   - Iterates over all files in the directory
   - Filters for JPG, JPEG, and PNG files (skips subdirectories)
   - Collects all matching image paths into a list
   - Converts the sorted list of images into a single PDF using `img2pdf.convert()`
3. **If the path is a single file:**
   - Checks that the file is JPG, JPEG, or PNG
   - Converts it to a new PDF path
4. Refuses to overwrite an existing PDF.

## Project Structure

```
Convert_a_image_to_pdf/
├── convert_image_to_pdf.py   # Main conversion script
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency versions
└── README.md
```

## Setup & Installation

```bash
cd Convert_a_image_to_pdf
uv sync
```

## How to Run

### Convert a single image

```bash
uv run python convert_image_to_pdf.py path/to/image.jpg
```

### Convert all supported images in a directory

```bash
uv run python convert_image_to_pdf.py path/to/image_directory/
```

By default, the output is named after the input (`image.pdf` or `directory.pdf`) beside the input. Use `--output` to choose another new path.

## Configuration

No environment variables or config files required. Use `--output` to choose the PDF destination.

## Testing

No formal test suite present.

## Limitations

- Only JPG, JPEG, and PNG images are supported.
- Directory conversion processes only the specified directory, not subdirectories.
- Existing PDF outputs are not overwritten.
