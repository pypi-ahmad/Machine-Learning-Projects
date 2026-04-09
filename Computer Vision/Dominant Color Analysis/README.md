# Dominant Color Finder

> A CLI script that finds the dominant color in an image by analyzing pixel value frequency using NumPy.

## Overview

This script reads an image from a user-provided path, flattens the pixel data, counts the occurrences of each unique pixel value, and identifies the top 3 most frequent values. It then creates and displays solid color swatches representing the dominant tone and single-channel color.

## Features

- Reads an image from a user-specified file path
- Displays the original image in an OpenCV window ("img")
- Analyzes pixel value frequency using `numpy.unique()` with counts
- Identifies the **top 3 most frequent pixel values**
- Displays a 300×300 color swatch for the dominant tone (top 3 values as BGR)
- Displays a 300×300 grayscale swatch for the single most dominant value
- Prints the dominant tone and color values to the console

## Project Structure

```
Dominant_color/
├── find-color.py
├── requirements.txt
├── shot.png            # Sample screenshot
└── README.md
```

## Requirements

- Python 3.x
- `opencv-python==4.3.0.36`
- `numpy==1.19.1`

## Installation

```bash
cd Dominant_color
pip install -r requirements.txt
```

## Usage

```bash
python find-color.py
```

When prompted, enter the full path to an image:

```
Enter Path :- /path/to/image.jpg
```

The script will print the top 3 values and display three OpenCV windows: "img" (original image), "Tone", and "color".

## How It Works

1. Reads the image path from user input, loads it with `cv2.imread()`, and displays it in an "img" window.
2. Converts the image to a NumPy array and uses `np.unique()` with `return_counts=True` to count occurrences of each pixel intensity value.
3. Sorts by frequency (descending) and selects the top 3 values.
4. Creates a 300×300 image filled with the top 3 values as a BGR tuple for the "Tone" swatch.
5. Creates a 300×300 grayscale image using only the single most frequent value for the "color" swatch.
6. Displays both swatches using `cv2.imshow()`.

## Configuration

No configuration files. The image path is provided interactively at runtime.

## Limitations

- Analyzes **individual channel values** rather than full RGB/BGR tuples — the "dominant color" is based on the most frequent single-byte value across all channels, not the most frequent pixel color.
- The top 3 values are used directly as a BGR tuple, which may not represent a meaningful color.
- No `cv2.waitKey()` call — the display windows may close immediately on some systems.
- No error handling beyond a basic `try/except` that prints "Path not found" and exits.
- Only works with images OpenCV can read; no format validation.
- Requires a GUI environment for `cv2.imshow()`.

## License

Not specified.
