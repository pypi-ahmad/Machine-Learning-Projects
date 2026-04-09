# QR Code Generator

> A Python script that generates a QR code image from a URL using the `qrcode` library.

## Overview

This script takes a hardcoded URL, generates a QR code with customizable appearance settings, and saves it as a PNG image file.

## Features

- Generates QR codes from any URL
- Configurable QR code version, error correction level, box size, and border
- Custom fill and background colors (default: red on white)
- Saves output as a PNG image
- Prints the encoded data list to console

## Project Structure

```
Qr_code_generator/
├── generate_qrcode.py    # Main script
├── url_qrcode.png        # Generated QR code output image
└── README.md
```

## Requirements

- Python 3.x
- `qrcode`
- `Pillow` (required by `qrcode` for image generation)

## Installation

```bash
cd "Qr_code_generator"
pip install qrcode[pil]
```

## Usage

1. Edit `generate_qrcode.py` and change the `input_URL` variable to your desired URL
2. Run the script:

```bash
python generate_qrcode.py
```

3. The QR code is saved as `url_qrcode.png` in the same directory.

## How It Works

1. Creates a `qrcode.QRCode` object with:
   - `version=1` (21×21 matrix, smallest size)
   - `error_correction=ERROR_CORRECT_L` (~7% error recovery)
   - `box_size=15` (15 pixels per QR box)
   - `border=4` (4-box quiet zone border)
2. Adds the URL data with `qr.add_data()`
3. Calls `qr.make(fit=True)` to auto-adjust the version if data exceeds capacity
4. Generates the image with red fill color and white background
5. Saves to `url_qrcode.png`

## Configuration

- `input_URL` — the URL to encode (currently `"https://www.google.com/"`)
- `version` — QR code version (1–40); higher = more data capacity
- `error_correction` — error correction level (`ERROR_CORRECT_L`, `M`, `Q`, or `H`)
- `box_size` — pixel size of each QR code box
- `border` — width of the quiet zone border in boxes
- `fill_color` / `back_color` — QR code colors (currently `"red"` / `"white"`)
- Output filename is hardcoded as `"url_qrcode.png"`

## Limitations

- URL is hardcoded; no command-line argument or user input support
- Output filename is hardcoded; overwrites the existing file each run
- No validation of the input URL
- Only generates PNG format

## Security Notes

No security concerns identified.

## License

Not specified.
