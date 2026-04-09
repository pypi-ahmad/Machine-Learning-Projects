# Create AudioBook in Python

## Overview

A Python script that reads a PDF file and converts its text content into spoken audio using text-to-speech. It reads each page aloud and also attempts to save the output as an MP3 file.

**Type:** CLI Utility

## Features

- Reads all pages from a PDF file using PyPDF2
- Converts extracted text to speech using pyttsx3 (offline text-to-speech engine)
- Iterates through every page of the PDF and speaks the content aloud
- Attempts to save the last page's text as an audio file (`audio.mp3`)

## Dependencies

- `PyPDF2` — for reading and extracting text from PDF files
- `pyttsx3` — for text-to-speech conversion (offline, no internet required)

Install with:

```bash
pip install PyPDF2 pyttsx3
```

## How It Works

1. Opens a PDF file named `file.pdf` in binary read mode using `PyPDF2.PdfFileReader`.
2. Initializes the pyttsx3 text-to-speech engine with `pyttsx3.init()`.
3. Loops through all pages of the PDF (`pdfReader.numPages`):
   - Extracts text from each page using `getPage(page_num).extractText()`.
   - Passes the text to `speaker.say(text)` and calls `speaker.runAndWait()` to speak it aloud.
4. After the loop, calls `speaker.stop()`.
5. Attempts to save the audio to `audio.mp3` using `engine.save_to_file()` — note: this references an undefined variable `engine` instead of `speaker`.

## Project Structure

```
Create AudioBook in Python/
├── Create Audio Book in Python.py    # Main script
└── README.md
```

## Setup & Installation

```bash
pip install PyPDF2 pyttsx3
```

A PDF file named `file.pdf` must be placed in the same directory as the script.

## How to Run

```bash
python "Create Audio Book in Python.py"
```

The script will read each page of `file.pdf` aloud through the system's default audio output.

## Testing

No formal test suite present.

## Limitations

- The PDF filename is hardcoded to `file.pdf`.
- The `save_to_file` call at the end references `engine` which is never defined — it should be `speaker`. This will cause a `NameError` at runtime.
- `save_to_file` is called with only the text from the last page, not the entire document.
- Uses deprecated PyPDF2 API (`PdfFileReader`, `getPage`, `numPages`, `extractText`) — modern PyPDF2 (v3+) uses `PdfReader`, `pages[]`, and `extract_text()`.
- PDF text extraction quality depends on the PDF structure; scanned PDFs (image-based) will produce no text.
- No command-line arguments for specifying input/output files.
