# Instagram Liker Bot

> A Selenium bot that auto-likes all posts on a specified Instagram user's profile.

## Overview

This script automates the process of liking Instagram posts. It logs into Instagram, navigates to a target user's profile via the search bar, opens the first post, and continuously likes and navigates to the next post until no more posts are available.

## Features

- Automated Instagram login with password hidden via `getpass`
- Search for any Instagram user by username
- Auto-likes every post on the target profile sequentially
- Automatically closes the post modal when all posts are liked
- ChromeDriver auto-management via `webdriver_manager`

## Project Structure

```
Instagram Liker Bot/
├── Instagram_Liker_Bot.py   # Main bot script
└── readMe.md
```

## Requirements

- Python 3.x
- `selenium`
- `webdriver-manager`
- Google Chrome browser

## Installation

```bash
cd "Instagram Liker Bot"
pip install selenium webdriver-manager
```

## Usage

```bash
python Instagram_Liker_Bot.py
```

1. A Chrome window opens Instagram's login page.
2. Enter your username when prompted.
3. Enter your password (hidden input via `getpass`).
4. Enter the target user's username to search for.
5. The bot opens the first post and likes all posts sequentially.

## How It Works

1. **Login**: Opens `instagram.com`, enters credentials (uses XPath to find login form fields), and clicks the login button.
2. **Search**: Locates the search bar via XPath, types the target username, and presses ENTER twice to navigate.
3. **Like loop**: Clicks the first post, then enters an infinite loop that:
   - Finds and clicks the like button
   - Finds and clicks the next arrow button
   - Waits 5 seconds between posts
4. **Exit**: When an exception occurs (no more next button), it finds and clicks the close button to exit the post modal.

## Configuration

- **Sleep timers**: 10-second waits after page loads, 7-second wait for search results, 5-second delay between likes.
- **XPaths**: All element selectors are hardcoded XPaths specific to Instagram's DOM at the time of writing.

## Limitations

- Hardcoded XPaths will break when Instagram updates its UI.
- Uses bare `except` blocks — any error triggers the exit routine, even non-related failures.
- No rate limiting beyond 5-second delays — Instagram may detect and block rapid liking.
- Does not handle the "Action Blocked" dialog Instagram shows for suspicious activity.
- Only works on public profiles or profiles the logged-in user follows.
- The search relies on pressing ENTER twice, which may not reliably navigate to the correct profile.

## Security Notes

- Uses `getpass` for password input (good practice — password is hidden).
- **Instagram ToS violation**: Automated liking violates Instagram's Terms of Service and may result in account suspension.

## License

Not specified.
