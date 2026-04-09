# Random Password Generator

> Two Python scripts for generating random passwords — one fixed-length, one with a structured composition formula.

## Overview

This project contains two independent password generator scripts. The first (`python-password-generator.py`) generates a fixed-length 16-character password by sampling from all ASCII letters, digits, and punctuation. The second (`random_password_gen.py`) takes a user-specified length and creates a password with a structured 50/30/20 ratio of alphabetic, numeric, and special characters.

## Features

### `python-password-generator.py` (Fixed-Length Generator)
- Generates a 16-character password
- Uses `string.ascii_letters`, `string.digits`, and `string.punctuation`
- Samples without replacement using `random.sample()`

### `random_password_gen.py` (Structured Generator)
- User-specified password length via input prompt
- 50/30/20 composition: 50% alphabetic, 30% numeric, 20% special characters
- Random upper/lower case for alphabetic characters
- Final password is shuffled for randomness
- Special characters drawn from `@#$%&*`

## Project Structure

```
Random_password_generator/
├── python-password-generator.py    # Simple 16-char password generator
├── random_password_gen.py          # Structured password generator with ratio
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `random`, `string`, `math`)

## Installation

```bash
cd "Random_password_generator"
```

No additional packages need to be installed.

## Usage

### Simple Generator

```bash
python python-password-generator.py
```

Output: a single 16-character random password (e.g., `Kx!7bQ#2mR@9pLfZ`)

### Structured Generator

```bash
python random_password_gen.py
```

```
Enter Password Length: 12
aB3#kM7&nR2@
```

## How It Works

### `python-password-generator.py`
1. Concatenates `string.ascii_letters + string.digits + string.punctuation` into a single character pool.
2. Uses `random.sample(total, 16)` to pick 16 unique characters (no repeats).
3. Joins and prints the result.

### `random_password_gen.py`
1. Takes password length as user input.
2. Calculates character counts using the 50/30/20 formula:
   - Alphabetic: `pass_len // 2`
   - Numeric: `math.ceil(pass_len * 30 / 100)`
   - Special: remainder
3. `generate_pass()` function picks random characters from each pool: `abcdefghijklmnopqrstuvwxyz`, `0123456789`, `@#$%&*`.
4. Alphabetic characters have a 50% chance of being uppercased.
5. All characters are combined into a list, shuffled with `random.shuffle()`, and joined into a string.

## Configuration

- `python-password-generator.py`: Password length is hardcoded at 16
- `random_password_gen.py`: Password length is user-provided; special character set is `@#$%&*`

## Limitations

- `python-password-generator.py` uses `random.sample()` (sampling without replacement), so each character appears at most once; maximum password length is limited to the pool size
- `random_password_gen.py` has a small special character set (`@#$%&*`, only 6 characters)
- Neither script saves or copies passwords to clipboard
- `random_password_gen.py` has no input validation (non-integer input will crash with `ValueError`)
- Neither uses cryptographically secure random generation (`random` module, not `secrets`)
- `python-password-generator.py` has a fixed length with no user configuration

## Security Notes

- Both scripts use the `random` module, which is **not cryptographically secure**. For security-sensitive applications, use `secrets` instead.

## License

Not specified.
