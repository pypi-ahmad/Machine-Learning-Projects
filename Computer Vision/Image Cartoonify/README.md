# Easy Cartoonify

> An OpenCV-based script that applies cartoon-style filters to images, with automatic file search across directories.

## Overview

This script prompts the user for an image filename and a parent directory, searches for the file recursively, changes the working directory to the image's location, and applies one of two cartoon stylization effects using OpenCV's `cv2.stylization()` function.

## Features

- **Recursive file search**: Finds any image file by name within a specified directory tree using `os.walk()`
- **Automatic directory change**: Sets the working directory to the image's parent folder
- **Two cartoon styles**:
  - Style 1: `sigma_s=150, sigma_r=0.25` (smoother, more stylized)
  - Style 2: `sigma_s=60, sigma_r=0.5` (sharper, more detailed)
- Displays the cartoonified result in an OpenCV window

## Project Structure

```
Easy_cartoonify/
├── easy_cartoonify.py
└── README.md
```

## Requirements

- Python 3.x
- `opencv-python`

## Installation

```bash
cd Easy_cartoonify
pip install opencv-python
```

## Usage

```bash
python easy_cartoonify.py
```

Interactive prompts:

```
Please enter the name of the image file that you want to process:    photo.jpg
Please enter the directory that may contain the image:    /home/user/Pictures
This script currently has 2 sytles. Please type 1 or 2.   1
```

The cartoonified image is displayed in an OpenCV window. Press any key to close.

## How It Works

1. **`find_the_image(file_name, directory_name)`** — Recursively walks the directory tree using `os.walk()`, collecting all matches. Returns the first match's full path.
2. Reads the image with `cv2.imread()`.
3. User selects style 1 or 2.
4. Applies `cv2.stylization()` with the chosen sigma parameters.
5. Displays the result with `cv2.imshow()` and waits for a key press.

## Configuration

- **Cartoon style parameters**:
  - Style 1: `sigma_s=150`, `sigma_r=0.25`
  - Style 2: `sigma_s=60`, `sigma_r=0.5`
- All parameters are hardcoded in the script.

## Limitations

- `find_the_image()` is called **twice** — once to print the path and once to read the image — causing redundant directory traversals.
- No validation that the style input is "1" or "2" — any other input prints "Invalid style selection" and exits.
- Will crash with `IndexError` if the file is not found (`files_found[0]` on an empty list).
- Does not save the output image — only displays it.
- Requires a GUI environment for `cv2.imshow()`.
- No support for batch processing or output directory specification.
- The `pathlib.Path` import is used but only for extracting the parent directory.

## License

Not specified.
