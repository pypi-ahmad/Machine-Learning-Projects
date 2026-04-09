# Recursive Password Generator

> A Python script that generates random passwords of a specified length using a recursive string-building approach.

## Overview

This script generates random passwords by recursively appending random printable characters until the desired length is reached. It runs in a continuous loop, prompting the user for a new password length each time, and exits when the user types `e`.

## Features

- Generates passwords of any user-specified length
- Uses all printable ASCII characters (`string.printable`) including letters, digits, punctuation, and whitespace
- Recursive character-by-character password construction
- Interactive loop — generate multiple passwords without restarting
- Type `e` to exit gracefully

## Project Structure

```
Recursive_password_generator/
├── generator.py       # Main script
├── Screenshot.png     # Screenshot of the program
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `random`, `string`)

## Installation

```bash
cd "Recursive_password_generator"
```

No additional packages need to be installed.

## Usage

```bash
python generator.py
```

```
 [?] Enter a length for your password (e for exit): 16
' Kx!7bQ#2mR@9pLf '

 [?] Enter a length for your password (e for exit): 8
' aB3#kM7& '

 [?] Enter a length for your password (e for exit): e
```

## How It Works

1. **`get_random_char()`**: Picks a random character from `string.printable` (100 characters: letters, digits, punctuation, whitespace).
2. **`stretch(text, maxlength)`**: Recursively appends a random character to `text` until `len(text) >= maxlength`, then returns the result.
3. **Main loop**: Continuously prompts for a password length. On valid integer input, calls `stretch('', maxlength)` and prints the result. On `'e'`, breaks out of the loop. On invalid input, prints an error message.

## Configuration

- Character pool is `string.printable` (all printable ASCII characters including whitespace and control-like characters like `\t`, `\n`, `\r`, etc.)
- No minimum or maximum length enforced

## Limitations

- Uses recursion, so very long passwords may hit Python's default recursion limit (~1000 characters)
- `string.printable` includes whitespace characters (`\t`, `\n`, `\r`, space, `\x0b`, `\x0c`) which may produce unexpected characters in passwords
- Bare `except` clause catches all exceptions, including `KeyboardInterrupt`
- Uses `random` module — not cryptographically secure
- No clipboard copy or file save functionality
- The password is printed with surrounding single quotes and spaces, which could cause confusion

## Security Notes

- Uses the `random` module, which is **not cryptographically secure**. For security-sensitive applications, use `secrets` instead.
- `string.printable` includes control-like whitespace characters that may not be accepted by all password fields.

## License

Not specified.
