# Unfollowers-Insta (bb8 Bot)

> A Selenium-based Instagram bot that identifies users you follow who don't follow you back.

## Overview

An automated Instagram bot (`InstaBot` class) that logs into your account using Selenium and ChromeDriver, navigates to your profile, scrapes your following and followers lists by scrolling through the pop-up dialogs, and then computes and prints the list of accounts that you follow but who don't follow you back.

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
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- selenium 3.141.0
- urllib3 1.25.10
- Google Chrome browser
- ChromeDriver (matching your Chrome version)

## Installation

```bash
cd "Unfollowers-Insta"
pip install -r requirements.txt
```

Download ChromeDriver from [https://chromedriver.chromium.org/](https://chromedriver.chromium.org/) and place it at `C:\Program Files (x86)\chromedriver.exe` (or update the path in the script).

## Usage

```bash
python insta_bot_bb8.py
```

1. Enter your Instagram username when prompted.
2. Enter your password (hidden input via `getpass`).
3. The bot opens Chrome, logs in, navigates to your profile.
4. It scrapes your following and followers lists.
5. Unfollower names are printed to the terminal.
6. The browser closes automatically.

## How It Works

1. **`__init__`:** Prompts for username/password, initializes ChromeDriver at a hardcoded path.
2. **`start`:** Navigates to `https://www.instagram.com/`.
3. **`login`:** Fills in credentials via XPath, clicks login, dismisses two dialog pop-ups ("Not Now" buttons).
4. **`open_profile`:** Clicks the profile link from the main page.
5. **`open_following` / `open_followers`:** Clicks the respective count links on the profile page.
6. **`get_following` / `get_followers`:** Calls `scroll_list()` and stores the resulting list of names.
7. **`scroll_list`:** Scrolls a pop-up scroll box to the bottom by repeatedly executing JavaScript `scrollTo`. Extracts account names from `<a>` tags. Closes the dialog.
8. **`get_unfollowers`:** Computes `[x for x in following if x not in followers]` and prints each name.
9. **`close`:** Quits the Chrome WebDriver.

## Configuration

- **ChromeDriver path:** Hardcoded as `C:\Program Files (x86)\chromedriver.exe` in `__init__`. Update this to match your system.
- **XPaths:** All UI element selectors are hardcoded XPaths that depend on Instagram's DOM structure.

## Limitations

- XPaths are fragile and will break when Instagram updates its frontend.
- Uses deprecated Selenium methods (`find_element_by_xpath`, `find_element_by_tag_name`) — these were removed in Selenium 4.
- Hardcoded `time.sleep()` delays instead of explicit waits.
- No error handling for login failures, network issues, or element-not-found errors.
- Only works on Windows with Chrome (hardcoded ChromeDriver path).
- The comparison uses a list, not a set, so performance degrades with large follower counts.

## Security Notes

- Credentials are entered at runtime and not stored, but they are sent through Selenium to Instagram's login form.
- No two-factor authentication support.

## License

Not specified.
