# PyQt5 Password Generator

> A GUI-based password generator built with PyQt5 that creates passwords from user-specified characters and saves them to a file.

## Overview

This application provides a PyQt5 desktop interface where users can specify a set of characters and a desired password length. It generates random passwords from those characters and logs them to a `passwords.txt` file. It also includes a confirmation dialog system and the ability to delete all saved passwords.

## Features

- PyQt5 GUI with fixed 500×500 window and a lock icon
- User-configurable character set via comma-separated input (default: `a,b,c,d`)
- User-configurable password length (default: 5)
- Random password generation from the specified character pool
- Passwords automatically logged to `passwords.txt` using Python's `logging` module
- Confirmation message box after generation (can be permanently dismissed)
- "Delete" button to clear all saved passwords with a confirmation dialog
- Persistent "show message" preference stored in a `showMessage` file

## Project Structure

```
PyQt5_Password_generator/
└── password_generator/
    ├── main.py          # PyQt5 GUI application
    ├── random_pass.py   # Password generation logic
    ├── passwords.txt    # Log file for generated passwords
    ├── showMessage      # Flag file ("1" = show confirmation, "0" = don't)
    └── lock.png         # Window icon image
```

## Requirements

- Python 3.x
- `PyQt5`

## Installation

```bash
cd "PyQt5_Password_generator/password_generator"
pip install PyQt5
```

## Usage

```bash
python main.py
```

1. Enter comma-separated characters in the "characters" field (e.g., `a,b,c,d,1,2,3`)
2. Enter the desired password length in the "length" field
3. Click **"generate password"** to create a password
4. The password is saved to `passwords.txt`
5. A confirmation dialog asks if you want to see this message again (click "No" to suppress future dialogs)
6. Click **"Delete"** to clear all saved passwords (with confirmation prompt)

## How It Works

1. **`random_pass.py`**: Contains `randCahr(chars)` (selects a random character via `random.choice()`) and `randomPass(chars, passLen)` which builds a password by calling `randCahr()` for each position up to `passLen`.
2. **`main.py`**: Creates a `QMainWindow` with input fields for characters and length. On "generate password" click, it splits the character input by commas, calls `randomPass()`, and logs the result to `passwords.txt` via `logging.info()`.
3. **Message Suppression**: The `showMessage` file stores `"1"` or `"0"`. On startup, it's read to determine whether to show the confirmation dialog. Clicking "No" writes `"0"` to suppress it.
4. **Delete**: Opens a `QMessageBox` with Yes/Cancel; on Yes, truncates `passwords.txt`.

## Configuration

- **`showMessage` file**: Set to `1` to show confirmation dialogs, `0` to suppress
- **`lock.png`**: Window icon; must be in the same directory as `main.py`
- **Default characters**: `a,b,c,d` (editable in the GUI)
- **Default length**: `5` (editable in the GUI)

## Limitations

- Passwords are logged in plaintext to `passwords.txt`
- The character input must be comma-separated; no validation for empty or duplicate characters
- No clipboard copy functionality
- No password strength indicator
- The `logging` module appends to the file on each run, but the "delete" function only clears via `open("passwords.txt", "w")`
- The `showMessage` file uses a bare `"1"` / `"0"` string — no error handling if the file is missing or corrupted
- `int()` conversion of password length has no error handling (will crash on non-numeric input)

## Security Notes

- Generated passwords are stored in **plaintext** in `passwords.txt`
- The `logging` module writes passwords directly to the log file with no encryption
- A bare `except` clause in `deletePasswords()` catches all exceptions

## License

Not specified.
