# Instagram Follow / Not-Follow Checker

> A Selenium bot that identifies Instagram users who don't follow you back.

## Overview

This script logs into your Instagram account, scrapes your Following and Followers lists by scrolling through the modal dialogs, then compares them to produce a table of users who you follow but who don't follow you back.

## Features

- Automated Instagram login via Selenium
- Scrapes full Following and Followers lists by auto-scrolling
- Computes the list of non-followers (people you follow who don't follow back)
- Displays results in a formatted table using PrettyTable
- Interactive credential input

## Project Structure

```
Instagram Follow- NotFollow/
├── main.py      # Main bot script with InstaBot class
└── README.md
```

## Requirements

- Python 3.x
- `selenium`
- `prettytable`
- Google Chrome browser
- [ChromeDriver](https://chromedriver.storage.googleapis.com/index.html) matching your Chrome version

## Installation

```bash
cd "Instagram Follow- NotFollow"
pip install selenium prettytable
```

Download ChromeDriver and place it in your system PATH.

## Usage

```bash
python main.py
```

1. Enter your Instagram username and password when prompted.
2. The bot logs in, navigates to your profile, and scrapes your Following and Followers lists.
3. A table of non-followers is printed to the console.

## How It Works

1. **`InstaBot.__init__`**: Opens Chrome, navigates to `instagram.com`, fills credentials, clicks login, and dismisses notification popups via "Not Now" buttons.
2. **`get_unfollowers()`**: Clicks on the Following link, calls `_get_names()` to scrape all following usernames, then clicks the Followers link and scrapes those. Computes the difference: users in Following but not in Followers.
3. **`_get_names()`**: Locates the scrollable modal dialog, scrolls it to the bottom repeatedly until `scrollHeight` stops changing, then extracts all `<a>` tag text values (usernames). Closes the modal after scraping.

## Configuration

- **ChromeDriver**: Uses default `webdriver.Chrome()` — ChromeDriver must be in PATH.
- **Sleep timers**: Various `sleep()` calls (1–4 seconds) between UI interactions.
- **Modal XPaths**: Hardcoded to `/html/body/div[5]/div/div/div[2]` and similar — specific to Instagram's DOM at the time of writing.

## Limitations

- Hardcoded XPaths will break when Instagram updates its HTML structure.
- ChromeDriver must be manually downloaded and placed in PATH (no `webdriver_manager`).
- No headless mode — a visible Chrome window is required.
- For accounts with very large follower/following lists, scrolling may be slow or timeout.
- The `datetime` import and `start` variable are unused.
- Instagram credentials entered in plaintext.

## Security Notes

- **Credentials entered in plaintext**: Password is visible in the terminal via `input()`. Consider using `getpass`.
- **Instagram ToS violation**: Automated scraping violates Instagram's Terms of Service and may lead to account restrictions.

## License

Not specified.
