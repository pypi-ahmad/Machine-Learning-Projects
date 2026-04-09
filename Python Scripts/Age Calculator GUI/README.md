# Age Calculator GUI

A desktop GUI application built with Tkinter that calculates a person's age based on their date of birth.

## Overview

- Provides a simple graphical interface where the user enters their name, birth year, month, and day, then clicks a button to see their calculated age
- **Project type:** GUI (Tkinter)

## Features

- Text entry fields for name, birth year, birth month, and birth day
- "Calculate age" button triggers age computation
- Calculates age in whole years, adjusting if the birthday has not yet occurred this calendar year
- Displays a personalized result label (`<Name>'s age is <age>.`) directly in the window
- Fixed-size, non-resizable window (280×300 pixels)

## Dependencies

| Package | Source |
|---------|--------|
| `tkinter` | Python standard library |
| `datetime` | Python standard library |

No external dependencies — standard library only.

## How It Works

1. A Tkinter root window is created (280×300, non-resizable) with the title "Age Calculator".
2. Four `Entry` widgets are laid out in a grid for Name, Year, Month, and Day, each with a corresponding `Label`.
3. A `Button` labeled "Calculate age" is bound to the `ageCalc()` function.
4. When clicked, `ageCalc()`:
   - Reads the year, month, and day entries and constructs a `datetime.date` object for the birth date.
   - Gets today's date via `date.today()`.
   - Subtracts the birth year from the current year, then decrements by 1 if the birthday has not yet occurred this year (comparing month and day).
   - Destroys any previous result label and creates a new `Label` displaying the result.
5. `root.mainloop()` keeps the window open.

## Project Structure

```
Age-Calculator-GUI/
├── age_calc_gui.py   # Main application script
└── README.md
```

## Setup & Installation

```bash
# No external packages required — uses only the Python standard library.
# If tkinter is not bundled with your Python installation:
pip install tk
```

> **Note:** On some Linux distributions, install Tkinter separately: `sudo apt install python3-tk`.

## How to Run

```bash
cd Age-Calculator-GUI
python age_calc_gui.py
```

A window will appear. Enter a name and birth date, then click **"Calculate age"**.

## Configuration

No configuration files or environment variables required.

## Testing

No formal test suite present.

## Limitations

- **No input validation** — non-numeric or out-of-range values for year, month, or day raise an unhandled `ValueError`.
- Empty fields are not handled gracefully (causes `ValueError`).
- Age is calculated in whole years only; months and days of age are not displayed.
- The window geometry is hardcoded to 280×300 and non-resizable.
- Previous result label is destroyed and recreated on each click rather than updated in place.
