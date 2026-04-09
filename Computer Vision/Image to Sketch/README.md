# Image to Sketch

> A menu-driven Python script that converts images to pencil sketches using OpenCV, with webcam capture support.

## Overview

This script can either capture an image from the webcam or accept an image file path, then converts the image into a pencil sketch through a series of OpenCV transformations (grayscale, inversion, Gaussian blur, and division blending). Intermediate images from each step are saved alongside the final sketch.

## Features

- Capture image from webcam or provide a file path
- Multi-step conversion with all intermediate images saved (gray, inverted, blurred, final)
- Output organized into named subfolders under `sketches/`
- Interactive menu-driven CLI interface
- Sample images included for testing

## Project Structure

```
ImageToSketch/
├── sketchScript.py                      # Main script
├── haarcascade_frontalface_default.xml  # Haar Cascade file (present but unused)
├── images-trial/
│   └── hulk.jpg                         # Sample input image
├── sketches/
│   └── hulk/                            # Example output folder
├── webcam/
│   └── data/                            # Webcam capture storage
└── README.md
```

## Requirements

- Python 3.x
- `opencv-python` (`cv2`)
- A webcam (if using capture option)

## Installation

```bash
cd ImageToSketch
pip install opencv-python
```

## Usage

```bash
python sketchScript.py
```

**Menu options:**
```
1 - Capture image from webcam.
2 - Input the image path.
3 - Exit
```

- **Option 1**: Captures a single frame from the webcam, saves it to `webcam/data/` with a timestamp filename, then prompts for an output name.
- **Option 2**: Prompts for an image file path, then prompts for an output name.
- **Option 3**: Exits the program.

After selecting an option, enter a name for the output. The sketch and intermediate images are saved to `sketches/<name>/`.

## How It Works

1. **Grayscale conversion**: `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`
2. **Inversion**: `255 - gray` to create a negative
3. **Gaussian Blur**: `cv2.GaussianBlur(inverted, (21, 21), 0)` on the inverted image
4. **Inverted Blur**: `255 - blurred`
5. **Pencil Sketch**: `cv2.divide(gray, inverted_blurred, scale=256.0)` — dividing the grayscale by the inverted blur produces the sketch effect

## Configuration

- **Output directory**: Hardcoded as `.\ImageToSketch\sketches` — must be adjusted based on working directory.
- **Webcam save path**: Hardcoded as `webcam\data\`.
- **Blur kernel size**: `(21, 21)` — adjusting this changes sketch detail level.

## Limitations

- Hardcoded Windows-style backslash paths (`webcam\\data\\`) — not cross-platform.
- The `haarcascade_frontalface_default.xml` file is present but never used in the code.
- Uses deprecated import syntax `from cv2 import cv2` which may fail on newer OpenCV versions.
- No input validation — entering a non-existent file path will crash.
- Webcam captures only a single frame (not a preview to choose the shot).
- `int(input(...))` will crash on non-numeric input.

## Security Notes

No security concerns identified.

## License

Not specified.
