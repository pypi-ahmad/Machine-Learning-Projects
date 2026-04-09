# Instagram Image Downloader

> A Selenium-based script that scrapes and downloads images from an Instagram profile page.

## Overview

This script automates Instagram login using Selenium, navigates to a specified public profile, parses the page HTML with BeautifulSoup to extract all image URLs, and downloads them to a local `insta/` folder.

## Features

- Automated Instagram login via Selenium
- HTML parsing with BeautifulSoup to extract all `<img>` tags
- Downloads all found images to a local directory
- Saves images as numbered JPG files

## Project Structure

```
instagram_image_downloader/
├── instagram.py       # Main scraper script
├── requirement.txt    # Dependencies
├── img/               # Asset folder
├── insta/             # Downloaded images output
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── README.md
```

## Requirements

- Python 3.x
- `selenium`
- `beautifulsoup4` (`bs4`)
- `requests`
- `pandas` (imported but not used in the current code)
- Google Chrome browser
- [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads) matching your Chrome version

### requirement.txt

```
selenium
time
bs4
requests
```

> **Note**: `time` is a Python standard library module and does not need to be listed in requirements.

## Installation

```bash
cd instagram_image_downloader
pip install selenium beautifulsoup4 requests pandas
```

Download ChromeDriver matching your Chrome version and place it in the project directory or system PATH.

## Usage

Before running, edit `instagram.py` to set:
- Your Instagram username (line 16: `username.send_keys('username')`)
- Your Instagram password (line 18: `password.send_keys('password')`)
- The target profile URL (line 26: `webdriver.get('https://www.instagram.com/virat.kohli/')`)

```bash
python instagram.py
```

Downloaded images will be saved to the `insta/` folder.

## How It Works

1. Opens Chrome via Selenium, navigates to Instagram login page, and waits 20 seconds for the page to load.
2. Enters hardcoded username and password, clicks login, and dismisses the notification popup.
3. Navigates to the target profile URL and waits 10 seconds.
4. Parses the page source with BeautifulSoup and selects all `<img>` tags.
5. Extracts the `src` attribute from each image and downloads the image data via `requests.get()`.
6. Saves each image as `insta/<number>.jpg`.

## Configuration

- **Credentials**: Hardcoded in the script — must be edited before use.
- **Target profile**: Hardcoded URL (`https://www.instagram.com/virat.kohli/`).
- **Output directory**: Hardcoded as `insta/`.
- **Login wait time**: 20-second sleep, 10-second wait after navigation.

## Limitations

- Instagram credentials are hardcoded in plaintext.
- Only downloads images visible on the initial page load (no scrolling to load more).
- `pandas` is imported twice but never used.
- Element selectors use class names (`y3zKF`, `HoLwm`) that are Instagram-specific and will break on UI updates.
- `time` listed in `requirement.txt` is a standard library module.
- The `else` block references `f` which may not be defined if no images were found.
- Variable name `webdriver` shadows the imported `webdriver` module.

## Security Notes

- **Hardcoded credentials**: Username and password are stored in plaintext in the source code. Use environment variables or `getpass` instead.
- **Instagram ToS violation**: Automated scraping violates Instagram's Terms of Service.

## License

Not specified.
