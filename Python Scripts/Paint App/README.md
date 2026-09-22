# Paint App

A local Tkinter drawing application with pencil, eraser, line, rectangle, oval, fill, and text tools.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Choose a tool from the toolbar, set the foreground or background color with the swatches or palette, and drag on the canvas to draw. The text tool opens a prompt for the text to place.

## Files and behavior

- `main.py` is the maintained entrypoint.
- `paint.py` is a preserved legacy example and is not the documented launch target.
- **Save** exports the canvas as PostScript (`.ps`). It does not write image formats such as PNG or JPEG.
- **Clear** and **New** require confirmation before removing canvas items.
- The app runs locally and does not upload drawings or use a network connection.

## Limits

- The fill tool changes the closest canvas item's fill; it is not pixel-level flood fill.
- There is no undo, redo, raster-image import, or PNG/JPEG export.
