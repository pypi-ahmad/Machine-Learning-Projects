# Leap Year Checker

> A Python script that determines whether a given year is a leap year.

## Overview

This script takes a year as user input and checks whether it is a leap year using the standard Gregorian calendar rules. It prints the result to the console.

## Features

- Accepts a year as integer input from the user
- Implements the standard leap year algorithm (divisible by 4, not by 100, unless also by 400)
- Prints a clear message indicating whether the year is or is not a leap year

## Project Structure

```
Leap_Year_Checker/
└── leapyear.py
```

## Requirements

- Python 3.x
- No external dependencies

## Installation

No installation required — the script uses only Python built-ins.

```bash
cd "Leap_Year_Checker"
```

## Usage

```bash
python leapyear.py
```

**Example:**
```
Enter a year:- 2024
2024 is a leap year!!
```

```
Enter a year:- 1900
1900 is not a leap year!!
```

## How It Works

The script applies the Gregorian leap year rule:

1. If the year is divisible by 4 **and** not divisible by 100 → leap year.
2. If the year is divisible by 400 → leap year.
3. Otherwise → not a leap year.

This is implemented as: `((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0)`.

## Configuration

No configuration needed.

## Limitations

- No input validation — entering a non-integer value will raise a `ValueError`.
- No handling for negative years or year 0.
- The script only processes a single year per execution; there is no loop for multiple checks.

## Security Notes

No security concerns identified.

## License

Not specified.
