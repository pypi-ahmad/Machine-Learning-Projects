# Unfollowers-Insta (bb8 Bot)

> A Selenium-based Instagram bot that identifies users you follow who do not follow you back.

## Overview

This automated Instagram bot uses Selenium and ChromeDriver to log in, open your profile, scrape following and follower lists from the pop-up dialogs, and print accounts that do not follow you back.

## Features

- Automated Instagram login with username/password prompt
- Secure password input using `getpass` (password not shown while typing)
- Scrolls through the full following and followers lists automatically
- Computes the set difference to find non-mutual follows (unfollowers)
- Prints unfollower names to the terminal
- Object-oriented design with an `InstaBot` class

## Project Structure

```
Unfollowers-Insta/
├── insta_bot_bb8.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Requirements

- Python 3.13+
- `selenium`, managed by uv in `pyproject.toml`
- Google Chrome browser
- Internet access for Selenium Manager to obtain a compatible ChromeDriver when needed

## Installation

```bash
cd "Unfollowers-Insta"
uv sync
```

## Usage

```bash
uv run python insta_bot_bb8.py
```

Or pass the username directly:

```bash
uv run python insta_bot_bb8.py --username your_username
```

1. Enter your Instagram username when prompted.
2. Enter your password (hidden input via `getpass`).
3. The bot opens Chrome, logs in, navigates to your profile.
4. It scrapes your following and followers lists.
5. Unfollower names are printed to the terminal.
6. The browser closes automatically.

## How it works

1. **`__init__`:** Receives username/password and initializes Chrome through Selenium Manager.
2. **`start`:** Navigates to `https://www.instagram.com/`.
3. **`login`:** Fills in credentials via XPath and clicks login.
4. **`open_profile`:** Clicks the profile link from the main page.
5. **`open_following` / `open_followers`:** Clicks the respective count links on the profile page.
6. **`get_following` / `get_followers`:** Calls `scroll_list()` and stores the resulting list of names.
7. **`scroll_list`:** Scrolls a pop-up scroll box to the bottom by repeatedly executing JavaScript `scrollTo`. Extracts account names from `<a>` tags. Closes the dialog.
8. **`get_unfollowers`:** Uses a follower-name set to compute non-mutual follows efficiently.
9. **`close`:** Quits the Chrome WebDriver.

## Configuration

- **XPaths:** All UI element selectors are hardcoded XPaths that depend on Instagram's DOM structure.

## Limitations

- XPaths are fragile and will break when Instagram updates its frontend.
- Hardcoded `time.sleep()` delays instead of explicit waits.
- No error handling for login failures, network issues, or element-not-found errors.
- Only works with Chrome and current Instagram page layouts.
- The comparison uses a list, not a set, so performance degrades with large follower counts.

## Security Notes

- Credentials are entered at runtime and not stored, but they are sent through Selenium to Instagram's login form.
- No two-factor authentication support.

## License

Not specified.
