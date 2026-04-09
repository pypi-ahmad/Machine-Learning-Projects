# Instagram Scrapper in Python

> A script using Instaloader to scrape Instagram profile information, followers, followees, and download posts.

## Overview

This script uses the `instaloader` library to retrieve detailed profile information from Instagram (username, user ID, follower/following count, bio, posts), list followers and followees, and download all posts from a specified profile. The script includes both authenticated and interactive login methods.

## Features

- Fetch Instagram profile details (username, user ID, post count, followers, followees, bio, external URL)
- Retrieve full follower and followee username lists (requires login)
- Download all posts from a target profile into numbered target directories
- Supports both scripted login and interactive terminal login

## Project Structure

```
Instagram Scrapper in Python/
├── Instagram Scapper In Python.py   # Main script
└── README.md
```

## Requirements

- Python 3.x
- `instaloader`

## Installation

```bash
cd "Instagram Scrapper in Python"
pip install instaloader
```

## Usage

```bash
python "Instagram Scapper In Python.py"
```

> **Note**: The script contains `!pip install instaloader` (Jupyter/IPython syntax) which will cause a `SyntaxError` when run as a standard Python script. Remove or comment out that line first.

## How It Works

1. Creates an `Instaloader` instance and loads a profile using `Profile.from_username()`.
2. Prints profile metadata: username, user ID, media count, followers, followees, bio, and external URL.
3. Logs in (either via `bot.login()` with hardcoded credentials or `bot.interactive_login()` for terminal prompt).
4. Retrieves all follower and followee usernames using `profile.get_followers()` and `profile.get_followees()`.
5. Loads a second profile (`wwe` as example) and iterates through all posts using `profile.get_posts()`, downloading each with `bot.download_post()`.

## Configuration

- **Target profile**: Hardcoded as `'aman.kharwal'` for info scraping and `'wwe'` for post downloading.
- **Login credentials**: Hardcoded as `user="your username"`, `passwd="your password"` — must be replaced.

## Limitations

- Contains `!pip install instaloader` (Jupyter magic command) that will fail in standard Python.
- Target usernames and credentials are hardcoded — not parameterized.
- The script is written as a sequential notebook-style script, not a reusable module.
- Downloading all posts from large accounts will take significant time and storage.
- No error handling for private profiles, rate limits, or network failures.

## Security Notes

- **Hardcoded credential placeholders**: `user="your username"` and `passwd="your password"` must be replaced. Never commit real credentials to source control.
- **Instagram ToS**: Automated scraping and downloading may violate Instagram's Terms of Service.

## License

Not specified.
