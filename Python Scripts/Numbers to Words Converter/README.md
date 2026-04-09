# Convert Numbers to Words

## Overview

A Python CLI script that converts any integer (including negative numbers and very large numbers) into its English word representation.

**Type:** CLI Utility

## Features

- Converts integers to their English word form (e.g., `144` → `One hundred and forty four`)
- Supports negative numbers (prefixed with `(negative)`)
- Supports very large numbers up to nonillions (via scale words: thousand, million, billion, trillion, quadrillion, quintillion, sextillion, septillion, octillion, nonillion)
- Handles special cases for teens (eleven, twelve, thirteen, etc.)
- Interactive prompt loop — keeps asking for input until the user types `exit`
- Input validation with error handling for non-numeric input

## Dependencies

- Python 3.x (no external dependencies)

## How It Works

1. The `converter(n)` function takes a string representation of an integer.
2. If the number starts with `-`, it prepends `(negative)` and strips the sign.
3. The number string is zero-padded to a multiple of 3 digits, then split into groups of 3.
4. Each 3-digit group is processed:
   - Hundreds digit → `"<digit> hundred"`
   - Tens digit → handles teens (10–12 specially, 13–19 with suffix logic), and multiples of ten with suffixes like `-ty`
   - Ones digit → direct word lookup
5. Scale words (thousand, million, etc.) are appended based on the group's position.
6. Words are joined and the first letter is capitalized.

## Project Structure

```
Convert_numbers_to_word/
├── converter.py      # Main script with converter() function and interactive loop
├── Screenshot.png    # Sample screenshot of usage
└── README.md
```

## Setup & Installation

No installation required. Only a working Python 3 interpreter is needed.

## How to Run

```bash
python converter.py
```

Then enter numbers at the prompt:

```
Enter any number to convert it into words or 'exit' to stop: 12345
12345 --> Twelve thousand three hundred and forty five
Enter any number to convert it into words or 'exit' to stop: exit
```

## Testing

No formal test suite present.

## Limitations

- Only handles integers — decimal numbers are not supported.
- The word construction logic uses index-based suffix selection (e.g., `range(3, 6, 2)` for "thir"/"fif") which is not immediately intuitive.
- No support for ordinal numbers (first, second, third, etc.).
- Trailing/leading whitespace in the output is handled with `strip()` but edge cases may exist for certain number patterns.

