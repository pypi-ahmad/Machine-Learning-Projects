# Barcode Generator

Generate Code 128, Code 39, EAN, UPC, and ISBN barcodes from the terminal.

```powershell
cd "Python Scripts/Barcode Generator"
uv sync
uv run python main.py --code HELLO123 --type code128 --save barcode.svg
```

`python-barcode` creates scannable SVG output. The built-in text display is a
terminal preview only and is not a standards-compliant barcode.
