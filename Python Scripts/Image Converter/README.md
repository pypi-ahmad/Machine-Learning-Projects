# Convert Image Format

## Overview

A collection of Python scripts for converting image formats between JPG and PNG using the Pillow library. Includes a dynamic batch converter and individual single-image converters.

**Type:** CLI Utility

## Features

- **Batch conversion** (`convertDynamic.py`): Recursively walks the current directory tree, converting all `.jpg` files to `.png` and all `.png` files to `.jpg`
- **Single JPG to PNG** (`JPGtoPNG.py`): Converts a single hardcoded JPG image to PNG format
- **Single PNG to JPG** (`PNGtoJPG.py`): Converts a single hardcoded PNG image to JPG format
- `IOError` handling in the batch converter (`convertDynamic.py`) with graceful exit

## Dependencies

- `Pillow` — for image format conversion (listed in `requirements.txt`)

Install with:

```bash
pip install -r requirements.txt
```

## How It Works

1. **convertDynamic.py**: Uses `os.walk(".")` to traverse all files in the current directory and subdirectories. For each file, checks the extension — if `.jpg`, opens it with Pillow and saves as `.png` (and vice versa). Prints a message if a file is neither `.jpg` nor `.png`.
2. **JPGtoPNG.py**: Opens a hardcoded file (`naruto_first.jpg`), converts to RGB, and saves as `naruto.png`.
3. **PNGtoJPG.py**: Opens a hardcoded file (`naruto_first.png`), converts to RGB, and saves as `naruto.jpg`.

## Project Structure

```
convert_Imgs/
├── convertDynamic.py     # Batch converter (recursive, both directions)
├── JPGtoPNG.py           # Single JPG → PNG converter
├── PNGtoJPG.py           # Single PNG → JPG converter
├── requirements.txt      # Dependencies
├── naruto_first.jpg      # Sample input image (JPG)
├── naruto_first.png      # Sample input image (PNG)
├── naruto_last.jpg       # Sample output image (JPG)
├── naruto_last.png       # Sample output image (PNG)
└── README.md
```

## Setup & Installation

```bash
pip install Pillow
```

## How to Run

**Batch conversion (both directions):**
```bash
cd convert_Imgs
python convertDynamic.py
```
This converts all JPG images to PNG and all PNG images to JPG in the current directory tree.

**Single image conversion:**

Edit the filenames inside `JPGtoPNG.py` or `PNGtoJPG.py` to match your input file, then run:

```bash
python JPGtoPNG.py
python PNGtoJPG.py
```

## Testing

No formal test suite present.

## Limitations

- `JPGtoPNG.py` and `PNGtoJPG.py` have hardcoded filenames; they must be edited manually for different input files.
- `convertDynamic.py` replaces the extension in the filename using string replacement (e.g., `filename.replace('jpg', 'png')`), which could cause issues if `jpg` or `png` appears elsewhere in the filename.
- No output directory option — converted files are saved alongside the originals.
- The batch converter runs on the current working directory (`.`), not a configurable path.
