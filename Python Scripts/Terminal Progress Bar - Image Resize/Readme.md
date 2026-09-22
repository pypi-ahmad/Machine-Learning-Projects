# Terminal Progress Bar - Image Resize

Batch-resize image files while preserving aspect ratio and showing terminal progress.

## Setup

Requirements: Python 3.13+.

```powershell
cd "Terminal Progress Bar - Image Resize"
uv sync
```

## Usage

```powershell
uv run python "progress_bar_ with_images_resizing.py" "C:\photos" --width 800 --height 600
```

By default, resized files are saved under `resize/` inside the input directory. The tool preserves each supported image's filename and format.

Options:

- `--width`: Maximum output width. Required.
- `--height`: Maximum output height. Required.
- `--output PATH`: Another output directory.
- `--overwrite`: Replace an existing resized file. Without it, that file is skipped.

## Behavior

The tool processes directly contained `.bmp`, `.gif`, `.jpeg`, `.jpg`, `.png`, and `.webp` files. Pillow's `thumbnail()` resizes each image to fit the requested bounding box without changing aspect ratio. Unsupported files and individual conversion failures are skipped and reported; no artificial sleep delays are added.

Resizing can overwrite only when `--overwrite` is supplied. Keep original files until you have inspected the output.

## Project files

```text
Terminal Progress Bar - Image Resize/
├── progress_bar_ with_images_resizing.py
├── pyproject.toml
└── uv.lock
```
