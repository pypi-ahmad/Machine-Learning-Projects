# Snapshot of Given Website

> A command-line tool that captures a full-page screenshot of any website using Selenium and headless Chrome.

## Overview

This script takes a URL as a command-line argument, opens it in a headless Chrome browser via Selenium, dynamically resizes the browser window to match the full page dimensions, and saves a full-page screenshot as `screenshot.png` in the current directory.

## Features

- Full-page screenshot capture (scrollWidth × scrollHeight)
- Headless Chrome operation (no visible browser window)
- Dynamic window resizing to capture the entire page
- Command-line URL input via `sys.argv`
- Outputs a `screenshot.png` file in the working directory

## Project Structure

```
Snapshot_of_given_website/
├── requirements.txt
└── snapshot_of_given_website.py
```

## Requirements

- Python 3.x
- `selenium==3.141.0`
- `chromedriver-binary==85.0.4183.38.0`
- Google Chrome browser installed

## Installation

```bash
cd "Snapshot_of_given_website"
pip install -r requirements.txt
```

> **Note:** The `chromedriver-binary` version in `requirements.txt` (`85.0.4183.38.0`) must match your installed Chrome version. Update the version number accordingly.

## Usage

```bash
python snapshot_of_given_website.py <URL>
```

**Example:**

```bash
python snapshot_of_given_website.py https://www.example.com
```

If no URL is provided:

```
Usage: snapshot_of_given_website.py URL
```

The screenshot is saved as `screenshot.png` in the current working directory.

## How It Works

1. Parses the URL from `sys.argv[1]`
2. Creates a headless Chrome WebDriver instance using `chromedriver_binary`
3. Navigates to the provided URL with `driver.get(url)`
4. Executes JavaScript to determine the full page dimensions:
   - `document.body.scrollWidth` for width
   - `document.body.scrollHeight` for height
5. Resizes the browser window to match the full page size via `driver.set_window_size()`
6. Saves the screenshot with `driver.save_screenshot('screenshot.png')`
7. Quits the driver and prints "SUCCESS"

## Configuration

- **Output filename**: Hardcoded as `screenshot.png` — change the string in `driver.save_screenshot()` to customize
- **ChromeDriver version**: Must match your Chrome browser version — update `chromedriver-binary` version in `requirements.txt`

## Limitations

- Uses `selenium 3.141.0` and an older `chromedriver-binary` version — may not work with modern Chrome versions
- The `chromedriver_binary` import approach is deprecated in favor of Selenium Manager (Selenium 4+)
- Output filename is hardcoded — running twice overwrites the previous screenshot
- No timeout handling for slow-loading pages
- Does not wait for dynamic content (JavaScript rendering) to complete before capturing
- Error handling only catches missing URL argument (`IndexError`), not network or driver errors

## Security Notes

No security concerns.

## License

Not specified.
