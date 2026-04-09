# Terminal Progress Bar with Image Resizing

> Batch image resizer with a terminal progress bar, using Pillow for image processing and tqdm for progress display.

## Overview

This script takes a directory of images and a target size, then resizes all images to fit within the specified dimensions (preserving aspect ratio via thumbnailing). A `tqdm` progress bar is displayed in the terminal to show the resizing progress. Resized images are saved to a `resize/` subfolder within the input directory.

## Features

- Batch resizes all images in a specified directory
- Preserves aspect ratio using Pillow's `thumbnail()` method
- Displays a terminal progress bar via `tqdm`
- Creates a `resize/` output subfolder automatically
- Interactive input for directory path and target dimensions

## Project Structure

```
Terminal_progress_bar_with_images_resizing/
├── progress_bar_ with_images_resizing.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- `tqdm==4.48.2` (specified in requirements.txt)
- `Pillow` (listed as `PIL==1.1.6` in requirements.txt, but should be installed as `Pillow`)

## Installation

```bash
cd Terminal_progress_bar_with_images_resizing
pip install tqdm Pillow
```

> **Note:** The `requirements.txt` lists `PIL==1.1.6`, but the correct installable package is `Pillow`, not `PIL`.

## Usage

```bash
python "progress_bar_ with_images_resizing.py"
```

The script will prompt for:

1. **Path to images** — directory containing the images to resize
2. **Size Height, Width** — target dimensions as comma-separated integers

**Example:**

```
Enter Path to images : C:\Users\me\photos
Size Height , Width : 800,600
```

Output:

```
Resizing Images: 100%|██████████| 25/25 [00:02<00:00, 10.5it/s]
Resizing Completed!
```

Resized images are saved as `resize/<original_filename>.jpg` inside the input directory.

## How It Works

1. Prompts the user for an image directory path and target size (H, W)
2. Changes the working directory to the input path via `os.chdir()`
3. Lists all items in the directory; creates a `resize/` subfolder if absent
4. Iterates over all items with a `tqdm` progress bar
5. For each file, opens it with Pillow, calls `thumbnail(size, Image.ANTIALIAS)` to resize while preserving aspect ratio
6. Saves the resized image as JPEG in the `resize/` subfolder
7. Adds a 0.1-second sleep per file (for visual progress bar smoothness)

## Configuration

No configuration files. All parameters are provided interactively at runtime:
- Image directory path
- Target dimensions (height, width)

## Limitations

- `Image.ANTIALIAS` is deprecated in Pillow 10+ (replaced by `Image.LANCZOS`) — will cause an `AttributeError` on newer Pillow versions
- The output filename appends `.jpg` to the original filename (e.g., `photo.png` becomes `photo.png.jpg`)
- All output is saved as JPEG regardless of the original format
- Non-image files in the directory will cause exceptions (caught and printed, but not skipped cleanly)
- Uses `os.chdir()` which changes the global working directory
- The 0.1-second `sleep()` per image artificially slows down processing
- Directory listings may include subdirectories, which will fail when opened as images

## Security Notes

No security concerns identified.

## License

Not specified.
