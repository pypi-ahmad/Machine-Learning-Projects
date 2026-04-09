# Convert Image to PDF

## Overview

A command-line utility that converts JPEG images to PDF format using the `img2pdf` library. Supports converting both a single JPG file and all JPG files within a directory into a single PDF.

**Type:** CLI Utility

## Features

- Convert a single `.jpg` image to a PDF
- Convert all `.jpg` images in a directory into a single multi-page PDF
- Automatic detection of whether the input is a file or directory
- Lossless conversion (img2pdf does not re-encode the image)

## Dependencies

From `requirements.txt`:

| Package | Version |
|---------|---------|
| img2pdf | 0.4.0   |

Additional standard library imports: `sys`, `os`

## How It Works

1. The script reads a file or directory path from `sys.argv[1]`.
2. **If the path is a directory:**
   - Iterates over all files in the directory
   - Filters for files ending with `.jpg` (skips subdirectories)
   - Collects all matching image paths into a list
   - Converts the list of images into a single `output.pdf` using `img2pdf.convert()`
3. **If the path is a single file:**
   - Checks that the file ends with `.jpg`
   - Converts it directly to `output.pdf`
4. If the path is neither a file nor a directory, prints an error message.

## Project Structure

```
Convert_a_image_to_pdf/
├── convert_image_to_pdf.py   # Main conversion script
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup & Installation

```bash
cd Convert_a_image_to_pdf
pip install -r requirements.txt
```

## How to Run

### Convert a single image

```bash
python convert_image_to_pdf.py path/to/image.jpg
```

### Convert all JPG images in a directory

```bash
python convert_image_to_pdf.py path/to/image_directory/
```

The output PDF will be saved as `output.pdf` in the current working directory.

## Configuration

No environment variables or config files required. The input path is the only argument.

## Testing

No formal test suite present.

## Limitations

- Only supports `.jpg` files — other image formats (`.png`, `.jpeg`, `.bmp`, `.tiff`) are ignored.
- The output is always named `output.pdf` in the current working directory with no option to customize the output path or filename.
- No argument validation — running without a command-line argument raises an `IndexError`.
- No recursive directory scanning — only processes files in the top level of the specified directory.
- No control over page ordering when converting a directory — order depends on `os.listdir()` which is filesystem-dependent.
- The file extension check is case-sensitive (`.jpg` only, not `.JPG` or `.JPEG`).
