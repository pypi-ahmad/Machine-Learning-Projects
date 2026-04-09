# Convert PNG Images to ICO Format

## Overview

A Python project that converts PNG images to ICO (icon) format using Pillow, available as both a terminal-based script and a Tkinter GUI application.

**Type:** GUI / CLI Utility

## Features

- **Terminal mode** (`convert.py`): Converts a hardcoded PNG file to ICO in the same directory
- **GUI mode** (`convertUI.py`): Tkinter GUI with:
  - File browser dialog filtered to `.png` files only
  - "Save As" dialog with `.ico` as default extension
  - Error message popup if no file is selected before converting
  - Success message popup after conversion
  - Light blue themed canvas with styled blue buttons
  - Window title set to "PNG to ICO Converter"

## Dependencies

- `Pillow` — for image opening and format conversion (listed in `requirements.txt` as `Pillow==7.2.0`)
- `tkinter` — for the GUI (included with standard Python on most platforms)

Install with:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install Pillow
```

## How It Works

1. **Terminal mode** (`convert.py`): Opens `input.png` using `Image.open()`, then saves it as `output.ico` using `img.save()`. Pillow handles the format conversion based on the file extension.
2. **GUI mode** (`convertUI.py`): Creates a Tkinter window (500×350, light blue canvas). The "Import PNG File" button opens a file dialog filtered to `.png` files. The selected image is stored in a global variable. The "Convert PNG to ICO" button opens a save dialog defaulting to `.ico` extension and saves the image.

## Project Structure

```
convert_png_images_to_ico_format/
├── convert.py          # Terminal-based converter
├── convertUI.py        # Tkinter GUI converter
├── input.png           # Sample input image
├── output.ico          # Sample output icon
├── requirements.txt    # Dependencies (Pillow==7.2.0)
└── README.md
```

## Setup & Installation

```bash
pip install Pillow
```

## How to Run

**Terminal mode:**
1. Place your PNG image as `input.png` in the project folder.
2. Run:
   ```bash
   python convert.py
   ```
3. `output.ico` will be generated in the same folder.

**GUI mode:**
```bash
python convertUI.py
```
1. Click "Import PNG File" to browse for a PNG image.
2. Click "Convert PNG to ICO" to choose a save location.

## Testing

No formal test suite present.

## Limitations

- Terminal mode uses hardcoded filenames (`input.png` → `output.ico`).
- The GUI script uses `tk.filedialog` and `tk.messagebox` without explicitly importing them from `tkinter` (may cause `AttributeError` on some Python versions).
- ICO files have size constraints (typically 256×256 max); large PNG images may not convert correctly without resizing.
- No batch conversion support.
