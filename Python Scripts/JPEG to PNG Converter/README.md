# Convert JPEG to PNG

## Overview

A Python project that converts JPEG images to PNG in two modes: a terminal converter and a Tkinter GUI with file browsing and save dialogs.

**Type:** GUI / CLI Utility

## Features

- **Terminal mode** (`converter_terminal.py`): Converts a hardcoded JPEG file to PNG in the same directory
- **GUI mode** (`converter_GUI.py`): Tkinter GUI with:
  - File browser dialog to select any JPEG image
  - "Save As" dialog to choose output location and filename
  - Orange-themed canvas with styled buttons

## Dependencies

- `Pillow` (PIL) — for image opening and format conversion
- `tkinter` — for the GUI (included with standard Python on most platforms)

Install Pillow with uv:

```powershell
uv sync --no-config
```

## How it works

1. **Terminal mode**: Opens `input.jpeg` from the current folder using Pillow's `Image.open()`, then saves it as `output.png` using `im1.save()`.
2. **GUI mode**: Creates a Tkinter window with an "Import JPEG File" button that opens a file dialog (`filedialog.askopenfilename`). The selected image is stored in a global variable. A "Convert JPEG to PNG" button triggers `filedialog.asksaveasfilename` with `.png` as the default extension and saves the image.

## Project Structure

```
Convert_JPEG_to_PNG/
├── converter_GUI.py         # Tkinter GUI converter
├── converter_terminal.py    # Terminal-based converter
├── input.jpeg               # Sample input image
├── output.png               # Sample output image
└── README.md
```

## Setup & Installation

```powershell
uv sync --no-config
```

## How to Run

**Terminal mode:**
```powershell
uv run --no-config python converter_terminal.py input.jpeg output.png
```

**GUI mode:**
```powershell
uv run --no-config python converter_GUI.py
```
1. Click "Import JPEG File" to browse for a JPEG image.
2. Click "Convert JPEG to PNG" to choose a save location.

## Testing

No formal test suite present.

## Limitations

- Terminal mode uses hardcoded filenames (`input.jpeg` → `output.png`).
- GUI mode does not validate that the selected file is actually a JPEG.
- No batch conversion support.
- The GUI error handling uses `tk.messagebox` but does not import it explicitly in the code.


