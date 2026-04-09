# Insta-Bot Follow & Send Message

> A Selenium-based Instagram bot that auto-follows users and sends them direct messages.

## Overview

This script automates Instagram interactions using Selenium WebDriver. Given a comma-separated list of usernames, it logs into Instagram, navigates to each user's profile, follows them, and sends a direct message.

## Features

- Automated Instagram login via Selenium
- Batch follow multiple users (comma-separated input)
- Send direct messages to each followed user
- ChromeDriver auto-management via `webdriver_manager`

## Project Structure

```
Insta-Bot-Follow-SendMsg/
├── instabot.py   # Main bot script
└── README.md
```

## Requirements

- Python 3.x
- `selenium`
- `webdriver-manager`
- `openpyxl` (imported but not used in the current code)
- Google Chrome browser

## Installation

```bash
cd Insta-Bot-Follow-SendMsg
pip install selenium webdriver-manager openpyxl
```

## Usage

```bash
python instabot.py
```

1. A Chrome browser window opens Instagram.
2. Enter the target usernames (comma-separated) when prompted.
3. Enter your Instagram username and password.
4. The bot logs in, visits each profile, attempts to follow, opens DMs, and prompts you for a message to send to each user.

## How It Works

1. **Login**: Navigates to `instagram.com`, fills in username/password fields, and clicks the submit button.
2. **Follow loop**: For each username, navigates to `instagram.com/<username>/`, locates the Follow button via XPath, and clicks it.
3. **Message**: Locates the Message button by class name, clicks it, finds the textarea, types the message provided via `input()`, and sends with ENTER key.
4. **Error handling**: Uses bare `try/except: pass` blocks — if follow or message elements aren't found, the bot silently continues.

## Configuration

- **Wait timeout**: `WebDriverWait` is set to 120 seconds.
- **Sleep timers**: Various `time.sleep()` calls (2–10 seconds) between actions.
- All credentials and target usernames are provided interactively at runtime.

## Limitations

- Relies on specific CSS class names and XPaths (e.g., `_862NM`, `mt3GC`) that may break when Instagram updates its UI.
- `openpyxl` is imported but never used.
- Bare `except: pass` blocks hide all errors silently.
- The message prompt appears once per user inside the send loop, requiring manual input for each.
- Instagram credentials are entered in plaintext via `input()` (visible on screen).
- Instagram may detect and block automated behavior, potentially resulting in account suspension.

## Security Notes

- **Credentials entered in plaintext**: The password is visible in the terminal. Consider using `getpass` instead.
- **Instagram ToS violation**: Automated interaction with Instagram violates their Terms of Service and may result in account bans.

## License

Not specified.
