# Image Resizer

A Pillow-based command-line utility for resizing one image or every supported image in a directory. It can resize by percentage, fixed width or height, or a maximum side length while preserving the aspect ratio where applicable.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py photo.jpg --width 800
```

Other common commands:

```powershell
# Resize each supported image under a directory to 50%.
uv run --no-config python main.py images --percent 50

# Fit an image within a 1,024-pixel square.
uv run --no-config python main.py photo.jpg --max-side 1024 --output resized.jpg
```

Use `--height` instead of `--width` to set one dimension, or pass both to force exact dimensions. JPEG output accepts `--quality` from 1 to 100; it defaults to 90.

For a single input image, omitting `--output` writes a sibling file with the `_resized` suffix. The tool rejects an output path that would overwrite the source image.
