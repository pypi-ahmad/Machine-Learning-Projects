# Color Picker GUI

## Overview

Color Picker GUI is a local Tkinter application for selecting colors with RGB sliders or a six-digit hexadecimal value. It displays HEX, RGB, HSL, HSV, and CMYK representations and keeps an in-memory palette for the current session.

## Setup and run

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run python main.py
```

Use the RGB sliders or enter a value such as `#ff0000`, then select **Apply**. The copy buttons place a displayed value on the system clipboard.

## Limitations

Saved palette colors exist only until the window closes. The app does not write files or persist the palette.

## Verification

Run `uv run python -m py_compile main.py` to check syntax. Color-conversion helpers can be checked without opening the GUI.
