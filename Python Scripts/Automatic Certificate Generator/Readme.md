# Automatic Certificate Generator

A Python script that batch-generates personalized certificates by reading names from a CSV file and overlaying them onto a certificate template image.

## Overview

This is a **CLI utility** that uses Pillow (PIL) for image manipulation and Pandas for CSV parsing. It reads a list of names from `list.csv`, draws each name onto a `certificate.png` template image, and saves the resulting certificates as individual PNG files in a `pictures/` directory.

## Features

- Reads recipient names from a CSV file (`list.csv`) with a `name` column
- Draws each name onto a certificate template image at a fixed position
- Uses a TrueType font (`arial.ttf`) at size 60 for text rendering
- Saves each generated certificate as `pictures/<name>.png`
- Batch processes all names in a single run

## Dependencies

> *Inferred from imports (no `requirements.txt` present)*

- `Pillow` (PIL)
- `pandas`

## How It Works

1. The script reads `list.csv` using `pandas.read_csv()`, expecting a column named `name`.
2. It loads a TrueType font (`arial.ttf`) at size 60.
3. For each row in the CSV, it opens `certificate.png` as the template.
4. Using `ImageDraw.Draw`, it draws the name at position `(150, 250)` in black `(0, 0, 0)`.
5. The resulting image is saved as `pictures/<name>.png`.

## Project Structure

```
Automatic Certificate Generator/
├── main.py          # Certificate generation script
├── list.csv         # CSV file with recipient names (column: name)
└── Readme.md        # This file
```

**Required but not included:**
- `certificate.png` — The certificate template image
- `arial.ttf` — The TrueType font file
- `pictures/` — Output directory (must exist before running)

## Setup & Installation

```bash
pip install Pillow pandas
```

1. Place a `certificate.png` template image in the project folder.
2. Place `arial.ttf` (or another TrueType font, updating the script accordingly) in the project folder.
3. Create a `pictures/` directory for output:
   ```bash
   mkdir pictures
   ```
4. Populate `list.csv` with names:
   ```csv
   name
   John Doe
   Jane Smith
   ```

## How to Run

```bash
cd "Automatic Certificate Generator"
python main.py
```

Generated certificates will be saved in the `pictures/` folder.

## Configuration

- **Text position:** Hardcoded at `(150, 250)` in `main.py` — adjust `xy` parameter to match your template layout.
- **Font and size:** Hardcoded to `arial.ttf` at size 60 — change in the `ImageFont.truetype()` call.
- **Text color:** Hardcoded to black `(0, 0, 0)` — change the `fill` parameter.
- **CSV file:** Must be named `list.csv` with a `name` column.

## Testing

No formal test suite present.

## Limitations

- The `certificate.png` template and `arial.ttf` font file are not included in the repository.
- The `pictures/` output directory must exist before running; the script does not create it automatically (the `os` module is imported but unused).
- Text position, font, and size are hardcoded — different certificate templates will require manual adjustment.
- No error handling for missing files (`certificate.png`, `arial.ttf`, `list.csv`).
- Output filenames are based on the `name` column value, so duplicate names will overwrite each other.
