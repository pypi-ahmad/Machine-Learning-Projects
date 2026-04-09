# Automatic FB Login

A Selenium-based script that automates logging into Facebook using Chrome WebDriver.

## Overview

This is a **CLI automation script** that uses Selenium WebDriver to open Facebook in a Chrome browser, fill in the user's credentials, and click the login button.

## Features

- Opens Facebook login page in a Chrome browser via Selenium
- Prompts the user for their Facebook user ID and password
- Automatically fills in the email and password fields
- Clicks the login button to authenticate

## Dependencies

> *Inferred from imports (no `requirements.txt` present)*

- `selenium`

## How It Works

1. The script prompts the user for their Facebook user ID and password via `input()`.
2. It initializes a Chrome WebDriver using a hardcoded path to `chromedriver.exe`.
3. The browser navigates to `https://www.facebook.com/`.
4. The script locates the email field (by ID `email`), password field (by ID `pass`), and login button (by ID `u_0_b`) using `find_element_by_id`.
5. It fills in the credentials and clicks the login button.

## Project Structure

```
Automatic FB login/
├── Project _ Automatic FB login.py   # Main login automation script
└── README.md                         # This file
```

## Setup & Installation

```bash
pip install selenium
```

1. Download [ChromeDriver](https://chromedriver.chromium.org/downloads) matching your Chrome version.
2. Update the `cd` variable in the script to point to your `chromedriver.exe` path (default: `C:\webdrivers\chromedriver.exe`).

## How to Run

```bash
cd "Automatic FB login"
python "Project _ Automatic FB login.py"
```

You will be prompted for:
1. Your Facebook user ID (email/phone)
2. Your Facebook password

## Configuration

- **ChromeDriver path:** Hardcoded as `cd='C:\\webdrivers\\chromedriver.exe'` — must be updated to match your local path.
- **Login URL:** Hardcoded to `https://www.facebook.com/`.

## Testing

No formal test suite present.

## Limitations

- Uses deprecated Selenium methods (`find_element_by_id`) that are removed in Selenium 4+.
- The ChromeDriver path is hardcoded and Windows-specific (`C:\webdrivers\chromedriver.exe`).
- The login button ID (`u_0_b`) is a dynamically generated Facebook ID that will vary across sessions and may not work.
- Password is entered via `input()` (visible on screen) rather than `getpass`.
- The user's credentials are printed to the console via `print(user_id)` and `print(password)`.
- No error handling for incorrect credentials, missing ChromeDriver, or page load failures.
- Facebook may block or flag automated login attempts.

## Security Notes

- **Credentials are printed to stdout** — `print(user_id)` and `print(password)` expose the password in plaintext on the terminal.
- The password is collected via `input()` (not `getpass`), meaning it is visible as the user types.
- Credentials are stored in memory only and not persisted to disk by the script.
