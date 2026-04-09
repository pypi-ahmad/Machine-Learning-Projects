# Paint App

> A drawing application built with Python's Tkinter, supporting multiple drawing tools including pencil, line, arc, rectangle, oval, and text.

## Overview

This is a GUI-based paint application that allows users to draw on a canvas using various tools. It uses Tkinter's `Canvas` widget for rendering and provides a menu-based interface for switching between drawing tools.

## Features

- **Pencil tool**: Freehand drawing that follows mouse movement (default tool)
- **Line tool**: Draw straight lines between click and release points
- **Arc tool**: Draw arc shapes with a fixed 150° extent
- **Rectangle tool**: Draw filled rectangles (red fill, pink outline)
- **Oval tool**: Draw filled ovals (midnight blue fill, yellow outline)
- **Text tool**: Place hardcoded text ("helloooo!") at click position with custom font
- Menu bar with "Options" dropdown for tool selection
- Quit option accessible from the menu

## Project Structure

```
Paint-App/
└── paint.py
```

## Requirements

- Python 3.x
- `tkinter` (standard library, included with most Python installations)

No external dependencies required.

## Installation

```bash
cd "Paint-App"
```

No pip packages needed.

## Usage

```bash
python paint.py
```

1. The application opens a window with a blank canvas.
2. Use the **Options** menu at the top to select a drawing tool.
3. **Pencil**: Click and drag to draw freehand.
4. **Line/Arc/Rectangle/Oval**: Click to set the start point, release to set the end point.
5. **Text**: Click to place text at that position.
6. Use **Options > Quit** to exit.

## How It Works

The `PaintApp` class manages the drawing state:

- Tracks the current drawing tool, mouse button state, and cursor positions.
- Binds `<Motion>`, `<ButtonPress-1>`, and `<ButtonRelease-1>` events to the canvas.
- **Pencil**: Draws continuous line segments between previous and current mouse positions while the left button is held down.
- **Other tools**: Records start coordinates on button press and end coordinates on release, then draws the corresponding shape.
- Each shape type has its own method (`pencil_draw`, `line_draw`, `arc_draw`, `oval_draw`, `rect_draw`, `text_draw`).

## Configuration

Shape properties are hardcoded in the drawing methods:

- **Line**: green color, smooth rendering
- **Arc**: blue fill, 150° extent
- **Rectangle**: red fill, pink outline, width 2
- **Oval**: midnight blue fill, yellow outline, width 2
- **Text**: "helloooo!" in lightblue, Helvetica 20pt bold italic

## Limitations

- No color picker — all shape colors are hardcoded.
- No brush size adjustment.
- No undo/redo functionality.
- No save/export capability.
- Text content is hardcoded to "helloooo!" — no user input for text.
- Canvas size is not explicitly set (defaults to Tkinter's default).
- All non-pencil shape methods (`line_draw`, `arc_draw`, `oval_draw`, `rect_draw`) pass coordinates in a potentially incorrect order (`x1, x2, y1, y2` instead of `x1, y1, x2, y2`).
- No eraser tool.

## Security Notes

No security concerns identified.

## License

Not specified.
