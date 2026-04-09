# Password Manager GUI

> A Tkinter-based password manager that stores website credentials in a local SQLite database, protected by a master password passed via command-line argument.

## Overview

This application provides a graphical interface for storing and retrieving website credentials (website, username, password). Data is persisted in a local SQLite database (`passwordManager.db`). Viewing stored records requires entering the master password, which is set as a command-line argument when launching the app.

## Features

- **Add credentials**: Store website name, username, and password in SQLite database
- **View all records**: Display all stored credentials in a formatted table (requires master password)
- **Hide records**: Toggle visibility of stored credentials
- **Master password protection**: Viewing records requires authentication via the master password
- **Input validation**: Alerts user if any field is left empty when adding a password
- **Persistent storage**: Credentials survive application restarts via SQLite

## Project Structure

```
Password-Manager-GUI/
└── passwords.py
```

## Requirements

- Python 3.x
- `tkinter` (standard library)
- `sqlite3` (standard library)
- `sys` (standard library)

No external dependencies required.

## Installation

```bash
cd "Password-Manager-GUI"
```

No pip packages needed.

## Usage

Run with a master password as a command-line argument:

```bash
python passwords.py YourMasterPassword
```

Or:

```bash
python3 passwords.py YourMasterPassword
```

### Adding a Password

1. Enter the website, username, and password in the respective fields.
2. Click **"Add Password"**.
3. A confirmation dialog appears on success.

### Viewing Passwords

1. Click **"Show All"**.
2. Enter the master password when prompted.
3. Records are displayed in the purple frame at the bottom.
4. Click **"Hide Records"** to clear the display.
5. After hiding, the button text changes to **"Show Records"** for subsequent views (not "Show All").

## How It Works

1. On startup, reads `sys.argv[1]` as the master password.
2. Connects to (or creates) `passwordManager.db` SQLite database.
3. Creates a `passwords` table with columns: `website`, `username`, `pass`.
4. **Submit**: Inserts a new record using parameterized SQL queries.
5. **Query**: Prompts for the master password via `simpledialog.askstring()`, then fetches all records if authentication succeeds.
6. **Hide**: Clears the display label and resets the button text.

## Configuration

- **Master password**: Set via command-line argument (`sys.argv[1]`).
- **Database file**: `passwordManager.db` (created in the current working directory).
- **Window size**: Fixed at 600×400 pixels.

## Limitations

- Master password is passed as a plaintext command-line argument (visible in process listings and shell history).
- Stored passwords are saved in **plaintext** in the SQLite database — no encryption.
- No option to delete or update individual records.
- No search/filter functionality.
- The master password is not stored or hashed — it's only held in memory from `sys.argv[1]`.
- Crashes if no command-line argument is provided (`IndexError` on `sys.argv[1]`).
- The query display uses fixed-width column formatting, which may clip long values.

## Security Notes

- **Plaintext password storage**: All credentials are stored unencrypted in the SQLite database.
- **Master password exposure**: Passed via command-line argument — visible in shell history, process lists, and task managers.
- **No password hashing**: The master password is compared as a plain string.
- **No input sanitization** beyond parameterized SQL queries (which do prevent SQL injection).

## License

Not specified.
