# PDF & Image Reader with Voice

> Python scripts that extract text from PDF files and images using OCR (Tesseract), then read the text aloud using text-to-speech (pyttsx3).

## Overview

This project contains two independent scripts: one for reading text from PDF files and another for reading text from images. Both use Tesseract OCR for text extraction and pyttsx3 for text-to-speech output. Extracted text is also saved to a `remember.txt` file.

## Features

- **PDF text extraction**: Converts PDF pages to JPEG images, then applies OCR to extract text
- **Image text extraction**: Directly applies OCR to image files
- **Text-to-speech**: Reads extracted text aloud using the Windows SAPI5 speech engine
- **Text persistence**: Saves extracted text to `remember.txt` for later reference
- Supports multi-page PDF files

## Project Structure

```
PDF Reader with Voice/
├── Image Reader/
│   ├── img2.jpg          # Sample image file
│   ├── ocr.py            # Image text extractor with voice
│   └── remember.txt      # Output text file
├── PDF Reader/
│   ├── PDF Reader.py     # PDF text extractor with voice
│   ├── remember.txt      # Output text file
│   └── sample.pdf        # Sample PDF file
└── License
```

## Requirements

- Python 3.x
- `pytesseract` — Python binding for Tesseract OCR
- `Pillow` (PIL) — Image processing
- `pyttsx3` — Text-to-speech engine
- `wand` — Python binding for ImageMagick (PDF Reader only)
- `speech_recognition` — Imported but unused in PDF Reader
- **External software**:
  - [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) must be installed
  - [ImageMagick](https://imagemagick.org/script/download.php) must be installed (for PDF Reader)

## Installation

```bash
cd "PDF Reader with Voice"
pip install pytesseract Pillow pyttsx3 wand SpeechRecognition
```

### External Dependencies

- **Tesseract OCR**: Install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (Windows) or via package manager on Linux/Mac.
- **ImageMagick**: Install from [imagemagick.org](https://imagemagick.org/script/download.php) (required for PDF conversion).

## Usage

### PDF Reader

```bash
cd "PDF Reader"
python "PDF Reader.py"
```

Reads `sample.pdf` in the current directory, extracts text via OCR, speaks it aloud, and saves to `remember.txt`.

### Image Reader

```bash
cd "Image Reader"
python ocr.py
```

Reads `img2.jpg` in the current directory, extracts text via OCR, speaks it aloud, and saves to `remember.txt`.

## How It Works

### PDF Reader (`PDF Reader.py`)

1. Opens a PDF file using `wand` (ImageMagick) at 300 DPI resolution.
2. Converts each page to a JPEG image blob.
3. Opens each JPEG blob with PIL and runs Tesseract OCR (`image_to_string`).
4. Collects recognized text from all pages.
5. Prints the text, speaks it via pyttsx3, and writes it to `remember.txt`.

### Image Reader (`ocr.py`)

1. Opens an image file (hardcoded as `img2.jpg`) using PIL.
2. Runs Tesseract OCR on the image.
3. Prints the text, writes it to `remember.txt`, and speaks it via pyttsx3.

## Configuration

- **Tesseract path**: Hardcoded in both scripts as:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\tesseract.exe"
  ```
  Change this to match your Tesseract installation path.
- **Input files**: Hardcoded — `sample.pdf` for PDF Reader, `img2.jpg` for Image Reader.
- **Voice engine**: Uses Windows SAPI5 with `voices[0]` (first available voice).

## Limitations

- Input filenames are hardcoded — no command-line arguments or file picker.
- Tesseract path is hardcoded to a Windows-specific location.
- Voice engine is Windows-only (`sapi5`).
- `speech_recognition` is imported in PDF Reader but never used.
- `os` is imported in Image Reader (`ocr.py`) but never used.
- PDF Reader only processes the last page's text for the `recognized_text` variable (overwrites in loop, though `speak()` uses the last `text` value).
- No error handling for missing files, Tesseract not found, or ImageMagick not installed.
- OCR accuracy depends on image/PDF quality.

## Security Notes

No security concerns identified.

## License

A `License` file is included in the project directory.
