# Password Manager

> A Tkinter-based password generator and manager that creates random passwords of varying strength and saves credentials to a local text file.

## Overview

This GUI application generates random passwords with selectable strength levels (low, medium, strong) and configurable length (8–32 characters). Users can enter a username and website, then save the generated credentials to a local `info.txt` file. Passwords can be copied to the clipboard.

## Features

- **Password generation** with three strength levels:
  - **Low**: Lowercase letters only (`a-z`)
  - **Medium**: Mixed case letters (`a-zA-Z`)
  - **Strong**: Mixed case, digits, and special characters (`a-zA-Z0-9 !@#$%^&*()`)
- **Configurable length**: 8 to 32 characters via dropdown
- **Copy to clipboard**: Uses `pyperclip` to copy the generated password
- **Save credentials**: Appends username, password, and website to `info.txt`
- **View saved passwords**: Prints all stored credentials to the console
- **Auto-creates** `info.txt` if it doesn't exist

## Project Structure

```
PASSWORD_MANAGER/
├── password.py    # Main application
└── info.txt       # Stored credentials (auto-created)
```

## Requirements

- Python 3.x
- `pyperclip`
- `tkinter` (standard library)
- `os` (standard library)
- `random` (standard library)

## Installation

```bash
cd "PASSWORD_MANAGER"
pip install pyperclip
```

## Usage

```bash
python password.py
```

1. **Select strength**: Choose Low, Medium, or Strong via radio buttons.
2. **Select length**: Pick a length (8–32) from the dropdown.
3. Click **"Generate"** to create a password.
4. Click **"Copy"** to copy the password to the clipboard.
5. Enter a **username** and **website** in the respective fields.
6. Click **"Save"** to append the credentials to `info.txt`.
7. Click **"Show all passwords"** to print saved credentials to the console.

## How It Works

1. **`checkExistence()`**: Checks if `info.txt` exists; creates it if not.
2. **`low()`**: Generates a random password string based on the selected strength radio button value (`var`) and length combo box value (`var1`).
3. **`generate()`**: Calls `low()` and inserts the result into the password entry field.
4. **`copy1()`**: Reads the entry field and copies content to clipboard via `pyperclip.copy()`.
5. **`appendNew()`**: Reads username, website, and password fields, formats them with separators, and appends to `info.txt`.
6. **`readPasswords()`**: Opens `info.txt`, reads all content, and prints it to stdout.

## Configuration

No configuration files. All settings are selected through the GUI:

- **Strength**: Radio buttons (Low=1, Medium=0, Strong=3)
- **Length**: Combo box (8–32)

## Limitations

- Saved passwords are stored in **plaintext** in `info.txt`.
- "Show all passwords" only prints to the console — not displayed in the GUI.
- `appendNew()` opens the file a second time at the end but never writes or closes it properly (missing `()` on `file.close`).
- The function name `low()` is misleading — it handles all three strength levels.
- Radio button values are non-sequential (1, 0, 3) — value `2` is unused, so selecting nothing with default `IntVar()` value `0` defaults to "Medium".
- No master password protection.
- No delete or edit functionality for saved credentials.

## Security Notes

- **Plaintext credential storage**: All usernames, passwords, and websites are written to `info.txt` without encryption.
- **No access control**: Anyone with filesystem access can read `info.txt`.
- **Weak randomness**: Uses `random.choice()` from Python's `random` module, which is not cryptographically secure. For password generation, `secrets` module is recommended.

## License

Not specified.
