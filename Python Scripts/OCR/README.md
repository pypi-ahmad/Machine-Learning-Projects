# OCR Tool

A local Streamlit interface for extracting text from images with Tesseract OCR. It also includes static sample output for exploring the text-cleaning and statistics view when Tesseract is unavailable.

## Requirements

- Python 3.14 or later
- Tesseract OCR installed on Windows and available on `PATH` for image extraction

The uv project installs the Python integration (`pytesseract`) and Pillow. It does not install the Tesseract executable or language data.

## Run

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

Use the **Extract Text** tab to upload PNG, JPEG, BMP, or TIFF images after Tesseract is configured. Select a language only when its matching Tesseract language data is installed.

## Behavior and limits

- Uploaded image bytes are processed locally by the running app.
- The demo tab uses fixed sample text; it does not infer text from an image.
- OCR accuracy depends on image quality, language data, layout, and Tesseract configuration.
- The optional ASCII brightness preview is a visual aid, not an OCR fallback.
