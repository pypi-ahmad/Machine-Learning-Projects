# QR Code Generator

Generate a PNG QR code from text or a URL.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python generate_qrcode.py "https://example.com" --output .\example.png
```

The output defaults to `qrcode.png` in the current directory. Existing files at the chosen output path are overwritten. The tool runs locally and makes no network requests.
