# Decimal to Binary Converter and Vice Versa

## Overview

A CLI-based number base converter that converts decimal numbers to binary and binary numbers to decimal. Uses Python's built-in `bin()` and `int()` functions for the conversions.

**Type:** CLI Utility

## Features

- **Decimal to Binary:** Converts a user-supplied decimal integer to its binary representation
- **Binary to Decimal:** Converts a user-supplied binary string to its decimal equivalent
- Interactive menu for selecting conversion direction
- Input validation for menu selection (catches `ValueError` for out-of-range options)

## Dependencies

No external packages required. Uses only Python built-in functions.

## How It Works

1. The user is presented with a menu: `1. Decimal to binary` or `2. Binary to decimal`.
2. **Option 1:** The user enters a decimal integer. The script uses `bin(dec)[2:]` to convert it to binary (stripping the `0b` prefix) and prints the result.
3. **Option 2:** The user enters a binary string. The script uses `int(binary, 2)` to convert it to decimal and prints the result.
4. If the menu selection is not 1 or 2, a `ValueError` is raised and caught, prompting the user to choose a valid option.

## Project Structure

```
Decimal_to_binary_convertor_and_vice_versa/
├── decimal_to_binary.py   # Main script
├── output.png             # Screenshot of the application
└── README.md
```

## Setup & Installation

1. Ensure Python 3.x is installed on your system.
2. No additional packages need to be installed.

## How to Run

```bash
python decimal_to_binary.py
```

## Configuration

No configuration required.

## Testing

No formal test suite present.

## Limitations

- Only supports integer decimal inputs; floating-point numbers are not handled.
- No validation that the binary input string contains only `0` and `1` characters (an invalid binary string will raise an unhandled `ValueError` from `int(binary, 2)`).
- Single conversion per run; the script does not loop for multiple conversions.
- Negative numbers are accepted for decimal-to-binary but the output includes a `-` prefix (Python behavior of `bin()`).

