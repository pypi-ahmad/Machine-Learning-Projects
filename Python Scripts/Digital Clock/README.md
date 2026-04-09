# Digital Clock

## Overview

A GUI-based digital clock application built with Python's Tkinter library. Displays the current system time in 12-hour format (`HH:MM:SS AM/PM`) with a selectable Light or Dark theme via a menu bar.

**Type:** GUI Application

## Features

- Real-time clock display in 12-hour format with AM/PM indicator
- **Theme switching** via a menu bar:
  - **Dark theme:** Blue background (`#22478a`) with black text (default)
  - **Light theme:** White background with black text
- Large Calibri 40 bold font for readability
- Canvas-based layout with 400×140 pixel window
- Refreshes every 1000 milliseconds (1 second)

## Dependencies

- `tkinter` (Python standard library)
- `time` (Python standard library — `strftime`)

No external packages required.

## How It Works

1. Creates a Tkinter window titled "Digital-Clock" with a 400×140 canvas.
2. A frame is placed at relative position (0.1, 0.1) filling 80% of width and height.
3. A label within the frame displays the current time using `strftime('%I:%M:%S %p')`.
4. The `time()` function updates the label text and reschedules itself via `lbl.after(1000, time)`.
5. A menu bar with a "Theme" menu allows switching between Light and Dark themes.
6. Each theme function creates a new frame with the appropriate background color and starts its own update loop.

## Project Structure

```
Digital_clock/
├── digital_clock.py   # Main application script
├── Digital Clock.PNG   # Screenshot of the application
└── README.md
```

## Setup & Installation

1. Ensure Python 3.x is installed on your system.
2. No additional packages need to be installed.

## How to Run

```bash
python digital_clock.py
```

## Configuration

No external configuration. Theme can be changed at runtime via the Theme menu in the application.

## Testing

No formal test suite present.

## Limitations

- Each theme switch creates a new frame and label on top of the old one without destroying the previous widgets, which could accumulate widgets over many switches.
- The function name `time` shadows the `time` module import (though `strftime` is imported directly, so this works but is confusing).
- No date display, only time.
- Window size is fixed at 400×140 via the canvas; resizing the window does not scale the clock.
- No alarm, timer, or stopwatch functionality.



