# Digital Clock With GUI

## Overview

A GUI-based digital clock application built with Python's Tkinter library. Displays the current system time in `HH:MM:SS` format, updated every 200 milliseconds. Features a bold yellow background with a large font display.

**Type:** GUI Application

## Features

- Real-time digital clock display in `HH:MM:SS` (24-hour format)
- Large bold "Boulder" font at size 68 for high visibility
- Yellow background (`#f2e750`) with dark foreground text (`#363529`)
- Border width of 25 pixels around the clock label
- Resizable window (default size: 420×150 pixels)
- Refreshes every 200 milliseconds for smooth time updates

## Dependencies

- `tkinter` (Python standard library)
- `time` (Python standard library)

No external packages required.

## How It Works

1. Creates a Tkinter window titled "Digital Clock" with dimensions 420×150 pixels.
2. A `Label` widget is configured with the Boulder font at size 68, bold, with a yellow background and dark foreground.
3. The `digital_clock()` function reads the current time using `time.strftime("%H:%M:%S")` and updates the label text.
4. `label.after(200, digital_clock)` schedules the function to run again after 200 milliseconds, creating a continuous update loop.

## Project Structure

```
Digital Clock With GUI/
└── Digital Clock Gui.py   # Main application script
```

## Setup & Installation

1. Ensure Python 3.x is installed on your system.
2. No additional packages need to be installed.

## How to Run

```bash
python "Digital Clock Gui.py"
```

## Configuration

No external configuration. Visual properties can be modified directly in the script:

- `text_font` — Font family, size, and weight (default: `("Boulder", 68, 'bold')`)
- `background` — Background color (default: `"#f2e750"`)
- `foreground` — Text color (default: `"#363529"`)
- `border_width` — Label border width (default: `25`)

## Testing

No formal test suite present.

## Limitations

- Uses 24-hour time format only; no option to switch to 12-hour format.
- The "Boulder" font may not be available on all systems, which could cause Tkinter to fall back to a default font.
- No date display, only time.
- No alarm or timer functionality.
