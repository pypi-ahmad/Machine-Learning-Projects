# Facebook Autologin

> A Selenium-based Python script that automates Facebook login with Chrome WebDriver.

## Overview

This script uses Selenium WebDriver to open Facebook in Chrome, fill in login credentials, and click the login button.

## Features

- Opens Facebook in a Chrome browser via Selenium
- Automatically fills in email/username and password fields
- Clicks the login button programmatically

## Project Structure

```
Facebook-Autologin/
├── facebookAuto.py    # Main script for automated Facebook login
├── pyproject.toml     # Project metadata and dependencies
└── uv.lock            # Locked dependency versions
```

## Requirements

- Python 3.13+
- `selenium`, managed by uv in `pyproject.toml`
- Chrome browser installed
- Internet access for Selenium Manager to obtain a compatible ChromeDriver when needed

## Installation

```bash
cd "Facebook-Autologin"
uv sync
```

## Usage

1. Run the script:
   ```bash
   uv run python facebookAuto.py
   ```

2. Enter the email or username and password only when prompted. The password is not echoed.

## How it works

1. Initializes a Chrome WebDriver through Selenium Manager.
2. Navigates to `https://www.facebook.com`.
3. Locates the email field by ID (`email`) and sends the prompted email/username.
4. Locates the password field by ID (`pass`) and sends the configured password.
5. Locates the login button by name (`login`) and clicks it.

## Configuration

Pass the optional username from the command line, or enter it when prompted:

```bash
uv run python facebookAuto.py --username your.email@example.com
```

The password is always requested through a hidden prompt.

## Limitations

- No error handling for failed logins, missing elements, or network issues
- No support for two-factor authentication

## Security Notes

- Do not put credentials in source code or commit them to version control.
- Supplying a username through `--username` may expose it in shell history; use the interactive prompt when that matters.

## License

Not specified.
