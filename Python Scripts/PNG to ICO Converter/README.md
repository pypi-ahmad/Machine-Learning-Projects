# PNG to ICO Converter

Single-file PNG-to-ICO converter with command-line and Tkinter file-picker modes.

## Setup

Requirements: Python 3.13+.

```powershell
cd "PNG to ICO Converter"
uv sync
```

## Usage

Convert from the command line:

```powershell
uv run python convert.py input.png output.ico
```

Or open the file-picker interface:

```powershell
uv run python convert.py --gui
```

The converter writes common icon sizes from 16×16 through 256×256 when the source image supports them. It refuses to overwrite an existing output file.

## Behavior

1. Validates that the input exists and has a `.png` extension.
2. Validates that the output name uses `.ico`.
3. Uses Pillow to create a multi-resolution ICO file.
4. Reports file and image errors without replacing an existing icon.

This is a format conversion utility. It does not improve a low-resolution source image or guarantee visual quality at every icon size.

## Project files

```text
PNG to ICO Converter/
├── convert.py
├── input.png
├── output.ico
├── pyproject.toml
└── uv.lock
```
