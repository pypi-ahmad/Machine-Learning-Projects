# PyQt5 Password Generator

A local PyQt5 desktop app that generates passwords from a user-supplied character list.

## Setup

This legacy PyQt5 app uses Python 3.13 because its Windows Qt runtime does not support Python 3.14.

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python password_generator\main.py
```

Enter comma-separated characters, choose a length, then select **generate password**.

## Data and security

Generated passwords are saved in plaintext to `password_generator/passwords.txt`; treat that file as sensitive and do not commit or share it. The app can clear the file through its **Delete** button. Password character selection now uses Python's `secrets` module.

The app runs locally and makes no network requests.
