# Calendar GUI

## Overview

A graphical calendar viewer that displays the full calendar for any user-specified year. Built using Python's `tkinter` for the GUI and the `calendar` module for rendering.

**Type:** GUI Application

## Features

- Text entry field for specifying a year
- Displays the complete 12-month calendar for the entered year in a new window
- "Show Calendar" button to generate the calendar
- "CLOSE" button to exit the application
- Calendar output rendered in a monospaced font (`consolas`) for proper alignment

## Dependencies

No `requirements.txt` present. Dependencies inferred from imports:

| Package   | Source           |
|-----------|------------------|
| tkinter   | Python stdlib    |
| calendar  | Python stdlib    |

## How It Works

1. A main `tkinter` window (`gui`) is created with the title "CALENDAR", sized 250×250 with a "misty rose" background.
2. The user enters a year into the `Entry` widget and clicks "Show Calendar".
3. The `show_calendar()` function validates the year, opens a child window (550×600), and calls `calendar.calendar(year)` to generate the full-year text calendar.
4. The generated calendar text is displayed in a `Label` widget with monospaced formatting.
5. The "CLOSE" button closes the main application window.

## Project Structure

```
Calendar GUI/
├── Calendar_gui.py           # Main application script
├── Calendar for the year.png # Screenshot of calendar output
├── Input Calendar.png        # Screenshot of input window
└── README.md
```

## Setup & Installation

1. Install [uv](https://docs.astral.sh/uv/).
2. Run `uv sync` from this directory. `tkinter` and `calendar` are included with standard Python distributions.

## How to Run

```bash
uv run python Calendar_gui.py
```

A small window will appear prompting for a year. Enter a year (e.g., `2026`) and click "Show Calendar".

## Configuration

No external configuration, environment variables, or secrets required.

## Testing

Run a syntax check with `uv run python -m py_compile Calendar_gui.py`.

## Limitations

- No option to view a single month — always displays the full year.
