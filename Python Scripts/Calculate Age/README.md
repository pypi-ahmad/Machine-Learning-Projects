# Calculate Age

A Python CLI script that calculates and displays a person's age in years, months, and days based on their input age and the current system date.

## Overview

This is a **CLI utility** that takes a person's name and age (in years) as input, then calculates their approximate age in months and days using the current system date and Python's `time` and `calendar` modules.

## Features

- Accepts user's name and age as interactive input
- Calculates age in three units: years, months, and days
- Accounts for leap years when computing total days
- Uses the system's local time for month/day calculations
- Displays formatted output with all three age representations

## Dependencies

> *Standard library only (no external dependencies)*

- `time` (standard library)
- `calendar` (standard library — `isleap` function)

## How It Works

1. The script prompts for the user's name and age (in years) via `input()`.
2. It retrieves the current local time using `time.localtime()`.
3. **Months** are calculated as `(age × 12) + current_month`.
4. **Days** are calculated by:
   - Iterating from `(current_year - age)` to `current_year`, adding 366 for leap years and 365 otherwise.
   - Adding the days elapsed in the current year (summing days for each completed month, plus the current day of the month).
5. The `judge_leap_year()` function uses `calendar.isleap()` to check for leap years.
6. The `month_days()` function returns the number of days in a given month, accounting for leap years in February.
7. The result is printed as: `<name>'s age is X years or Y months or Z days`.

## Project Structure

```
Calculate_age/
├── calculate.py   # Main age calculation script
└── README.md      # This file
```

## Setup & Installation

No external packages required. Only Python 3 is needed.

```bash
# Verify Python is installed
python --version
```

## How to Run

```bash
cd Calculate_age
python calculate.py
```

You will be prompted for:
1. Your name
2. Your age (in whole years)

### Example Output

```
input your name: Alice
input your age: 25
Alice's age is 25 years or 303 months or 9131 days
```

## Configuration

No configuration files, environment variables, or external settings.

## Testing

No formal test suite present.

## Limitations

- The age input is expected as a whole number (integer years); non-integer or invalid input will cause an error.
- The calculation assumes the user's birthday is January 1st of `(current_year - age)`, so the day count is approximate.
- No input validation — entering a non-numeric age will crash with a `ValueError`.
- The month calculation adds the current month to `age × 12`, which gives months-of-life only if the birthday is in January.
- The `end=""` in the print statement produces output on a single line, which may not render correctly in all terminals if redirected.
