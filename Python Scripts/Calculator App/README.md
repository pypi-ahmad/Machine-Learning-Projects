# Calculator App

## Overview

A GUI calculator application built with Python's Tkinter library. Provides a standard calculator interface with number buttons, arithmetic operators, backspace, clear, and an equals button to evaluate expressions.

**Type:** GUI Application

## Features

- Number buttons 0–9 with grid layout
- Arithmetic operators: `+`, `-`, `*`, `/`, `^` (exponentiation via `**`)
- Decimal point (`.`) input
- Backspace (`<-`) button to delete the last character
- Clear (`C`) button to reset the input field
- Equals (`=`) button to evaluate the expression
- Division-by-zero error handling with a popup alert ("Cannot divide by 0 ! Enter valid values")
- Color-coded buttons: dark grey for numbers, orange for operators, light grey for utility buttons
- Non-resizable window
- Right-aligned entry field with custom font size 15

## Dependencies

- `tkinter` (Python standard library)
- `functools` (Python standard library)

No external packages required.

## How It Works

1. The `cal()` function creates the main Tkinter window and lays out all widgets using a grid layout.
2. Number and operator buttons append their respective characters to the entry field via `get_input()`.
3. The `calc()` function reads the entry field contents and evaluates them using Python's `eval()`.
4. If a `ZeroDivisionError` occurs, a popup window alerts the user.
5. The `backspace()` function removes the last character from the entry field.
6. The `clear()` function empties the entry field entirely.
7. `functools.partial` is used to create reusable button templates with consistent styling.

## Project Structure

```
Create_calculator_app/
├── calculator.py   # Main application script
├── output.png      # Screenshot of the application
└── README.md
```

## Setup & Installation

1. Ensure Python 3.x is installed on your system.
2. No additional packages need to be installed.

## How to Run

```bash
python calculator.py
```

## Configuration

No configuration required.

## Testing

No formal test suite present.

## Limitations

- Uses `eval()` to evaluate expressions, which can execute arbitrary Python code — this is a security concern if the input were ever sourced externally.
- No support for parentheses in expressions.
- No keyboard input support; all input must be via mouse clicks on buttons.
- The exponentiation button displays `^` but internally inserts `**`.
- No history of previous calculations.

## Security Notes

- The use of `eval()` on user input is inherently unsafe. In this context (local GUI with direct button input only), the risk is limited, but the entry field also accepts pasted text which could contain arbitrary Python code.




