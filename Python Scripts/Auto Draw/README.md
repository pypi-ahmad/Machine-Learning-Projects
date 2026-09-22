# Python Auto Draw

The script previews its movement plan by default. Add `--draw` only after the
target canvas is ready; moving the pointer to a screen corner triggers
PyAutoGUI's fail-safe.

> An automated spiral-drawing script that uses PyAutoGUI to draw a shrinking rectangular spiral on screen.

## Overview

This script uses the `pyautogui` library to control the mouse and draw a rectangular spiral. It gives the user 10 seconds to switch to a drawing application, then drags the mouse in a right-down-left-up pattern with decreasing distances until the spiral reaches the center.

## Features

- Automated mouse control to draw a rectangular spiral
- 10-second delay to switch to a drawing application
- Smooth mouse dragging with configurable duration per segment
- Spiral shrinks by 5 pixels per direction change until reaching zero

## Project Structure

```
Python_auto_draw/
├── python-auto-draw.py    # Main script
├── pyautoguidemo.gif      # Demo animation
└── README.md
```

## Requirements

- Python 3.x
- `pyautogui`

## Installation

```bash
cd "Python_auto_draw"
uv sync
```

## Usage

1. Open a drawing application (e.g., MS Paint) and select a drawing tool
2. Run the script:

```bash
uv run python python-auto-draw.py
```

3. Review the preview, then run `uv run python python-auto-draw.py --draw`
4. Switch to the drawing application during the configurable delay

## How it works

1. **Delay**: `time.sleep(10)` gives the user time to switch to a drawing application.
2. **Initial click**: `pyautogui.click()` activates the drawing tool at the current cursor position.
3. **Spiral loop**: Starting with `distance = 250`, the script:
   - Drags right by `distance` pixels
   - Decreases `distance` by 5
   - Drags down by `distance` pixels
   - Drags left by `distance` pixels
   - Decreases `distance` by 5
   - Drags up by `distance` pixels
4. Each drag has a `duration=0.1` seconds. The loop continues until `distance` reaches 0.

## Configuration

- `distance = 250` — initial spiral arm length (line 8); increase for a larger spiral
- `duration = 0.1` — speed of each drag segment in `pyautogui.dragRel()` calls
- `time.sleep(10)` — startup delay in seconds; adjust as needed
- Distance decrement is hardcoded at 5 pixels per step

## Limitations

- No way to stop the script once started (must kill the process)
- The starting position is wherever the mouse cursor happens to be
- The spiral always shrinks by exactly 5 pixels — not configurable without editing code
- Only draws a rectangular spiral; no other shapes
- Requires a drawing application to be open and focused with a tool selected

## Security Notes

No security concerns. The script controls the mouse, so be aware it will take over mouse input during execution.

## License

Not specified.
