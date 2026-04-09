# Image to Speech

> Extracts text from an image using OCR and converts it to speech using Google Text-to-Speech.

## Overview

This script uses Tesseract OCR (`pytesseract`) to extract text from an image, saves the extracted text to a file, then converts it to an MP3 audio file using Google Text-to-Speech (`gTTS`). The audio file is automatically opened for playback. The user can choose to keep or delete the generated files afterward.

## Features

- OCR text extraction from images via Tesseract
- Text-to-speech conversion using Google TTS (English)
- Automatic audio playback after generation
- Option to keep or delete generated text and audio files
- Supports any image format that PIL can open

## Project Structure

```
Imagetospeech/
├── image_to_speech.py   # Main script
└── Readme.md
```

## Requirements

- Python 3.x
- `pytesseract`
- `Pillow` (PIL)
- `gTTS`
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on the system

## Installation

```bash
cd Imagetospeech
pip install pytesseract Pillow gTTS
```

You must also install Tesseract OCR separately. The script will prompt you for the Tesseract executable path at runtime.

## Usage

```bash
python image_to_speech.py
```

1. Enter the path to the Tesseract executable (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`).
2. Enter the path to the image file.
3. The script extracts text, generates speech, and plays the audio.
4. When prompted, enter `Y` to delete generated files or `N` to keep them.

## How It Works

1. Sets `pytesseract.pytesseract.tesseract_cmd` to the user-provided path.
2. Opens the image with `PIL.Image.open()` and runs `pytesseract.image_to_string()`.
3. Writes the extracted text to `./Imagetospeech/text.txt`.
4. Reads the text file, replaces newlines with spaces, and passes it to `gTTS()` to generate `./Imagetospeech/imagetospeech.mp3`.
5. Opens the MP3 using `os.system("start ...")` (Windows-specific).
6. Prompts user to delete or keep the generated files.

## Configuration

- **Tesseract path**: Provided interactively at runtime via `input()`.
- **Language**: Hardcoded to English (`lang='en'`).
- **Output paths**: Hardcoded to `./Imagetospeech/text.txt` and `./Imagetospeech/imagetospeech.mp3`.

## Limitations

- Hardcoded file paths require running from a specific working directory.
- `os.system("start ...")` is Windows-only — will not work on macOS or Linux.
- No error handling for invalid image paths or missing Tesseract installation.
- Only supports English language for TTS.
- Input validation for the Y/N prompt does not handle unexpected input gracefully.

## Security Notes

No security concerns identified.

## License

Not specified.
