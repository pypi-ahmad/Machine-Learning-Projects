# Create a Simple Stopwatch

## Overview

A GUI-based stopwatch application built with Python's Tkinter library. The application displays elapsed time in `HH:MM:SS` format and provides Start, Stop, and Reset controls.

**Type:** GUI Application

## Features

- Displays elapsed time in `HH:MM:SS` format
- Start, Stop, and Reset buttons with state management (buttons are enabled/disabled contextually)
- Initial "Ready!" display before the stopwatch is started
- Reset works both while running and while stopped
- Fixed minimum window size of 250×70 pixels
- Uses Verdana 30 bold font for the time display

## Dependencies

- `tkinter` (Python standard library)
- `datetime` (Python standard library)

No external packages required.

## How It Works

1. A global counter tracks elapsed seconds starting from zero.
2. The `counter_label` function uses `label.after(1000, count)` to schedule a recursive call every 1000 milliseconds (1 second), incrementing the counter each tick.
3. The counter value is converted to `HH:MM:SS` format using `datetime.utcfromtimestamp()` and `strftime()`.
4. Button states are toggled: Start is disabled while running; Stop is disabled while stopped; Reset is disabled initially.
5. Resetting sets the counter back to zero and updates the display to `00:00:00`.

## Project Structure

```
Create_a_simple_stopwatch/
└── stopwatch.py       # Main application script
```

## Setup & Installation

1. Ensure Python 3.x is installed on your system.
2. No additional packages need to be installed (Tkinter is included with standard Python distributions).

## How to Run

```bash
python stopwatch.py
```

## Configuration

No configuration required.

## Testing

No formal test suite present.

## Limitations

- Time resolution is 1 second (not suitable for sub-second timing).
- The counter uses integer seconds and `datetime.utcfromtimestamp()`, so it will not correctly display times beyond 24 hours.
- The stopwatch does not persist state across application restarts.
- No lap or split time functionality.
