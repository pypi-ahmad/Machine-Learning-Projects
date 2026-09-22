# Image to Speech

Extract text from an image with local Tesseract OCR, save that text, and create an MP3 using Google Text-to-Speech.

## Requirements

Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) separately, then make `tesseract.exe` available on `PATH`. If it is installed elsewhere, pass its full path with `--tesseract`.

## Run

```powershell
uv sync --no-config
uv run --no-config python image_to_speech.py receipt.png
```

By default, this writes `receipt.txt` and `receipt.mp3` next to `receipt.png`. Select explicit destinations when needed:

```powershell
uv run --no-config python image_to_speech.py receipt.png `
  --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe" `
  --text-output output\receipt.txt `
  --audio-output output\receipt.mp3
```

Use `--language` to choose a gTTS language code and `--overwrite` only when replacing existing outputs is intended.

## Data boundary

OCR runs locally. Creating the MP3 sends the extracted text to Google Text-to-Speech, so it requires internet access. The script does not automatically play audio or delete generated files.
