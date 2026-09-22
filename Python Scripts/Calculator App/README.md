# Calculator App

## Overview

A GUI calculator built with Python's Tkinter library. It provides number buttons, arithmetic operators, backspace, clear, and an equals button for evaluating expressions.

**Type:** GUI Application

## Features

- Number buttons 0–9 with grid layout
- Arithmetic operators: `+`, `-`, `*`, `/`, `^` (exponentiation via `**`)
- Decimal point (`.`) input
- Backspace (`<-`) button to delete the last character
- Clear (`C`) button to reset the input field
- Equals (`=`) button to evaluate the expression
- Clear popup messages for division by zero and invalid expressions
- Color-coded buttons: dark grey for numbers, orange for operators, light grey for utility buttons
- Non-resizable window
- Right-aligned entry field with custom font size 15

## Dependencies

- `tkinter` (Python standard library)
- `functools` (Python standard library)

No external packages required.

## How it works

1. The `cal()` function creates the main Tkinter window and lays out all widgets using a grid layout.
2. Number and operator buttons append their respective characters to the entry field via `get_input()`.
3. The `calc()` function evaluates only numeric arithmetic using Python's `ast` module.
4. Invalid expressions and division by zero show a popup message without changing the current input.
5. The `backspace()` function removes the last character from the entry field.
6. The `clear()` function empties the entry field entirely.
7. `functools.partial` is used to create reusable button templates with consistent styling.

## Project Structure

```
Calculator App/
├── calculator.py   # Main application script
├── output.png      # Screenshot of the application
└── README.md
```

## Setup & Installation

1. Install [uv](https://docs.astral.sh/uv/).
2. Run `uv sync` from this directory.

## How to Run

```bash
uv run python calculator.py
```

## Configuration

No configuration required.

## Testing

Run a syntax check with `uv run python -m py_compile calculator.py`.

## Limitations

- There are no dedicated buttons for parentheses, although typed or pasted parentheses are supported.
- The Enter key is not wired to the equals button.
- The exponentiation button displays `^` but internally inserts `**`.
- No history of previous calculations.

## Security Notes

Expressions are parsed as arithmetic only. Names, function calls, attributes, and other Python syntax are rejected.




