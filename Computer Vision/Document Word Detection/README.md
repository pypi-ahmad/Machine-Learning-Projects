# Document Word Detection

> An OpenCV-based script that detects and highlights words/text regions in a document image using morphological operations and contour detection.

## Overview

This script performs OCR-style word detection on a document image. It uses OpenCV to remove horizontal and vertical lines, applies edge detection and dilation to identify text regions, then draws bounding rectangles around detected words. The result is displayed in multiple windows and saved as an output image.

## Features

- Reads and resizes a document image (`test.jpeg`) to 600×600
- Removes horizontal and vertical lines using morphological opening
- Applies Gaussian blur and Otsu thresholding for binarization
- Uses adaptive thresholding and Canny edge detection for text region identification
- Dilates edges to merge nearby text into word-level bounding boxes
- Draws orange bounding rectangles around detected text regions (area > 20 pixels)
- Displays intermediate processing steps (threshold, lines, edges, contours)
- Saves final output to `output.jpg`

## Project Structure

```
Document-Word-Detection/
├── Word_detection.py
└── README.md
```

## Requirements

- Python 3.x
- `opencv-python`
- `numpy`
- `imutils`

## Installation

```bash
cd Document-Word-Detection
pip install opencv-python numpy imutils
```

## Usage

Place a document image named `test.jpeg` in the project directory, then run:

```bash
python Word_detection.py
```

The script will display several windows showing processing stages. Press any key to close all windows. The final annotated image is saved as `output.jpg`.

## How It Works

1. **Preprocessing**: Reads `test.jpeg`, resizes to 600×600, converts to grayscale, applies Gaussian blur.
2. **Line removal**: Uses Otsu binarization, then applies morphological opening with horizontal (25×1) and vertical (1×25) kernels to detect and remove document lines by drawing white contours over them.
3. **Text detection**: Applies adaptive Gaussian thresholding on the line-removed image, runs auto Canny edge detection (via `imutils`), dilates edges with a 5×5 kernel.
4. **Bounding boxes**: Finds contours on the dilated image, filters by area (> 20 pixels), and draws orange rectangles around each detected region.
5. **Output**: Saves the annotated frame to `output.jpg`.

## Configuration

- **Input image**: Hardcoded as `test.jpeg` — must be placed in the project directory.
- **Resize dimensions**: Hardcoded to 600×600 pixels.
- **Morphological kernels**: Horizontal (25×1) and vertical (1×25) for line detection.
- **Minimum contour area**: 20 pixels (filters out noise).
- **Dilation kernel**: 5×5 with 1 iteration.

## Limitations

- Input filename is hardcoded to `test.jpeg` — no CLI arguments or file picker.
- Image is forcibly resized to 600×600, which distorts non-square images.
- No actual OCR/text recognition — only detects text *regions* visually.
- Requires a GUI environment for `cv2.imshow()` display windows.
- No error handling if `test.jpeg` is missing.
- Threshold and kernel parameters are tuned for specific document types and may not generalize well.

## License

Not specified.
