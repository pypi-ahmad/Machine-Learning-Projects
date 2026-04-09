# Reduce Image File Size

> Reduces an image's dimensions (and thus file size) by a configurable scale factor using OpenCV.

## Overview

A Python script that reads an input JPEG image, scales it down by dividing both width and height by a factor (default 5), displays the resized image briefly, and saves the result to a new file.

## Features

- Reads a JPEG image using OpenCV
- Resizes the image by a configurable integer scale factor (default `k = 5`)
- Uses `INTER_AREA` interpolation for high-quality downscaling
- Displays the resized image in a window for 500 milliseconds
- Saves the resized output to `resized_output_image.jpg`
- Prints original and resized image dimensions (height, width, channels)

## Project Structure

```
Reduce_image_file_size/
├── reduce_image_size.py       # Main script
├── input.jpg                  # Sample input image
├── resized_output_image.jpg   # Generated output image
└── README.md
```

## Requirements

- Python 3.x
- `opencv-python`

## Installation

```bash
cd "Reduce_image_file_size"
pip install opencv-python
```

## Usage

1. Place your image as `input.jpg` in the project folder.
2. Run the script:

```bash
python reduce_image_size.py
```

3. The resized image is saved as `resized_output_image.jpg` in the same folder.

## How It Works

1. Reads `input.jpg` using `cv2.imread()`.
2. Calculates new dimensions by dividing both width and height by the scale factor `k` (integer division).
3. Resizes using `cv2.resize()` with `INTER_AREA` interpolation (optimal for shrinking).
4. Displays the result in a window for 500 ms via `cv2.imshow()` and `cv2.waitKey(500)`.
5. Writes the output to `resized_output_image.jpg` using `cv2.imwrite()`.

## Configuration

- **Scale factor (`k`)**: Set on line 7 of `reduce_image_size.py`. Default is `5`, meaning the image is reduced to 1/5th of its original width and height.
- **Input filename**: Hardcoded as `input.jpg`.
- **Output filename**: Hardcoded as `resized_output_image.jpg`.

## Limitations

- Input and output filenames are hardcoded; no CLI arguments supported.
- Only processes a single image per run.
- The scale factor is an integer, limiting the granularity of resizing.
- No error handling if `input.jpg` is missing or unreadable.
- The display window timeout (500 ms) is hardcoded and may be too brief.
- Does not control JPEG compression quality — only resizes dimensions.

## Security Notes

No security concerns.

## License

Not specified.
