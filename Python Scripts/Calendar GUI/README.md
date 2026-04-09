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
3. The `showCal()` function opens a **new** `Tk()` window (550×600), retrieves the year from the entry field, and calls `calendar.calendar(find_year)` to generate the full-year text calendar.
4. The generated calendar text is displayed in a `Label` widget with monospaced formatting.
5. The "CLOSE" button calls `exit()` to terminate the application.

## Project Structure

```
Calendar GUI/
├── Calendar_gui.py           # Main application script
├── Calendar for the year.png # Screenshot of calendar output
├── Input Calendar.png        # Screenshot of input window
└── README.md
```

## Setup & Installation

1. Ensure Python 3.x is installed (tkinter and calendar are included with standard Python distributions).
2. Clone or download this repository.

```bash
pip install tk  # Only if tkinter is not bundled with your Python installation
```

## How to Run

```bash
cd "Calendar GUI"
python Calendar_gui.py
```

A small window will appear prompting for a year. Enter a year (e.g., `2026`) and click "Show Calendar".

## Configuration

No external configuration, environment variables, or secrets required.

## Testing

No formal test suite present.

## Limitations

- Creates a new `Tk()` instance for the calendar display instead of using a `Toplevel` widget, which can cause issues with multiple Tk root windows.
- No input validation — entering a non-integer value will cause a `ValueError` crash.
- The calendar window opens a second `mainloop()`, which is not recommended in tkinter.
- No option to view a single month — always displays the full year.
- The "CLOSE" button calls `exit()`, which terminates the entire Python process rather than just closing the window.
