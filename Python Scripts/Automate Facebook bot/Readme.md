# Automate Facebook Bot

A Selenium-based automation script that logs into Facebook and automates posting to multiple Facebook groups.

## Overview

This is a **CLI bot** that uses Selenium WebDriver (Chrome) to automate Facebook login and group posting. The user provides their Facebook credentials, a list of group IDs, and a message, and the script iterates through each group to trigger the post action.

## Features

- Automated Facebook login via Selenium WebDriver
- Accepts multiple Facebook group IDs (comma-separated) as input
- Iterates through each specified group to trigger the post action (currently posts the literal string `"message"` — see Limitations)
- Uses `webdriver_manager` for automatic ChromeDriver installation
- Secure password input via `getpass`

## Dependencies

> *Inferred from imports (no `requirements.txt` present)*

- `pyautogui`
- `selenium`
- `webdriver_manager`

## How It Works

1. The script prompts the user for comma-separated Facebook group IDs and a message to post.
2. A `FacebookLogin` class is instantiated with the user's email and password.
3. ChromeDriver is automatically installed/managed via `webdriver_manager.chrome.ChromeDriverManager`.
4. The script navigates to Facebook's login page, fills in email and password fields, and clicks the login button.
5. For each group ID, the script navigates to `https://facebook.com/groups/<id>`, clicks on the post composer area (identified by CSS class names), types the literal string `"message"` (bug — should use the `message` variable), and clicks the post button.
6. `time.sleep()` calls are used between actions to allow pages to load.

## Project Structure

```
Automate Facebook bot/
├── script.py       # Main automation script
└── Readme.md       # This file
```

## Setup & Installation

```bash
pip install pyautogui selenium webdriver-manager
```

Ensure Google Chrome is installed on your system.

## How to Run

```bash
cd "Automate Facebook bot"
python script.py
```

You will be prompted for:
1. Facebook group IDs (comma-separated)
2. The message to post
3. Your Facebook email
4. Your Facebook password (hidden input)

## Configuration

- `LOGIN_URL` is hardcoded to `https://www.facebook.com/login.php` in the script.
- ChromeDriver path is managed automatically by `webdriver_manager`.
- No config files or environment variables are used; all input is collected at runtime.

## Testing

No formal test suite present.

## Limitations

- Uses deprecated Selenium methods (`find_element_by_id`, `find_element_by_class_name`) that are removed in Selenium 4+.
- Facebook CSS class names (e.g., `a8c37x1j ni8dbmo4 stjgntxs l9j0dhe7`) are obfuscated and change frequently, making the selectors fragile.
- Relies on fixed `time.sleep()` delays (up to 45 seconds per group) rather than explicit waits.
- The `pyautogui` import is unused in the script.
- The message variable `"message"` is sent as a literal string instead of the user-provided `message` variable (line 51: `send_keys("message")` should be `send_keys(message)`).
- Facebook may block or flag automated login attempts.

## Security Notes

- Facebook email and password are collected at runtime via `input()` and `getpass()`. They are stored in memory only and not persisted to disk.
- Credentials are passed in plaintext to Selenium form fields.
- Facebook group IDs are entered as user input and not validated.
