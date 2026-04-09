# Image Watermark

> Batch-adds a watermark image to all PNG and JPG files in a specified folder.

## Overview

This script takes a folder of images and a watermark image path as input, then applies the watermark to the bottom-right corner of every `.png` and `.jpg` file in that folder. Watermarked images are saved to an `output/` subdirectory within the source folder.

## Features

- Batch processing of all PNG and JPG images in a folder
- Automatic watermark resizing to 8% of the base image width
- Watermark positioned in the bottom-right corner with 20px margin
- Output saved to an `output/` subdirectory (auto-created if missing)
- Preserves original image color mode (RGB or palette)
- Supports transparent watermarks via RGBA compositing

## Project Structure

```
Image_watermark/
├── watermark.py        # Main script
├── requirements.txt    # Dependencies
└── README.md
```

## Requirements

- Python 3.x
- `Pillow` (PIL)

### requirements.txt

```
PIL==1.1.6
```

> **Note**: The requirements.txt lists `PIL==1.1.6`, but the correct installable package is `Pillow`.

## Installation

```bash
cd Image_watermark
pip install Pillow
```

## Usage

```bash
python watermark.py
```

1. Enter the folder path containing images to watermark.
2. Enter the path to the watermark image (should be a PNG with transparency for best results).
3. Watermarked images are saved to `<folder>/output/`.

## How It Works

1. Prompts user for the image folder path and watermark image path.
2. Changes working directory to the image folder and lists all files.
3. For each `.png` or `.jpg` file:
   - Opens the base image and watermark image.
   - Resizes the watermark to 8% of the base image width (square aspect).
   - Calculates bottom-right position with 20px padding.
   - Creates a new transparent RGBA canvas, pastes the base image, then pastes the watermark with its alpha mask.
   - Converts back to the original image mode and saves to `output/` at 100% quality.

## Configuration

- **Watermark size**: Hardcoded to 8% of the base image width (`position[0] * 8 / 100`).
- **Position offset**: 20px from bottom-right corner.
- **Blur**: A box blur option is commented out in the code (`ImageFilter.BoxBlur(2)`).
- **Output quality**: Saved at `quality=100` with `optimize=True`.

## Limitations

- Watermark is resized to a square (same width and height), which may distort non-square watermark images.
- Only processes `.png` and `.jpg` files (case-sensitive — `.PNG` or `.JPG` are skipped).
- The `watermark.size` expression on its own line has no effect (return value is not stored).
- `ImageFilter` is imported but only used in a commented-out line.
- No error handling for invalid paths or unsupported image formats.

## Security Notes

No security concerns identified.

## License

Not specified.
