# Image Viewer GUI

A Tkinter image viewer with folder navigation, zoom, rotation, and fit-to-window controls. Pillow provides support for JPG, PNG, GIF, BMP, TIFF, and WebP images.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Open a file or folder from the toolbar. You can also open an image immediately:

```powershell
uv run --no-config python main.py "C:\path\to\photo.jpg"
```

Use the toolbar or arrow keys to move through images in the same folder. The plus, minus, fit, and rotate controls change only the display; source image files are never modified.
