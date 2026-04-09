# Calculator GUI

## Overview

A standard calculator application with a graphical user interface built using Python's `tkinter` library. Supports arithmetic operations (addition, subtraction, multiplication, division) via clickable buttons.

**Type:** GUI Application

## Features

- Addition, subtraction, multiplication, and division
- Decimal point input support
- Clear (AC) button to reset the display
- Division-by-zero handling (displays "Undefined")
- Non-resizable fixed-size window
- Right-aligned display with custom font styling (Calibri)

## Dependencies

No `requirements.txt` present. Dependencies inferred from imports:

| Package  | Source           |
|----------|------------------|
| tkinter  | Python stdlib    |

## How It Works

1. A `tkinter` root window is created with title "Standard Calculator" and resizing disabled.
2. An `Entry` widget serves as the display, right-aligned with a light cyan background.
3. Number buttons (0–9) and a decimal button call `buttonClick()`, which appends the pressed character to the display.
4. Operator buttons (+, -, x, /) call `buttonGet()`, which stores the first operand and operator type in global variables, then appends the operator symbol to the display.
5. The equals button calls `buttonEqual()`, which parses the second operand from the display string (splitting on the operator character), performs the calculation, and displays the result.
6. The AC button calls `buttonClear()` to clear the display.
7. All buttons are arranged in a grid layout mimicking a standard calculator.

## Project Structure

```
Calculator-GUI/
├── script.py      # Main application with GUI and calculator logic
└── README.md
```

## Setup & Installation

1. Ensure Python 3.x is installed (tkinter is included with standard Python distributions).
2. Clone or download this repository.

```bash
pip install tk  # Only if tkinter is not bundled with your Python installation
```

## How to Run

```bash
cd Calculator-GUI
python script.py
```

A calculator window will appear.

## Configuration

No external configuration, environment variables, or secrets required.

## Testing

No formal test suite present.

## Limitations

- Only supports two-operand expressions (e.g., `5+3`). Chaining operations (e.g., `5+3-2`) without pressing equals first will produce incorrect results.
- The operator character is appended visually to the display string; entering an operator after another operator may cause a `ValueError` crash.
- No keyboard input support — all input must be via mouse clicks.
- No parentheses or advanced mathematical functions.
- Input validation is minimal — only catches non-numeric first operand via `ValueError`.
- The `num1` and `math` global variables are used without initialization guards, so pressing `=` before entering an operator will raise a `NameError`.
