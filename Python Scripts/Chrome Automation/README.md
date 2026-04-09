# Chrome Automation

## Overview

A Python script that automatically opens a predefined list of websites in Google Chrome. Useful for quickly launching your daily set of frequently visited sites.

**Type:** CLI Utility

## Features

- Opens multiple URLs sequentially in Google Chrome
- Uses Python's built-in `webbrowser` module
- Prints each URL to the console as it is opened
- Preconfigured with a set of common developer/productivity sites

## Dependencies

No `requirements.txt` present. Dependencies inferred from imports:

| Package     | Source           |
|-------------|------------------|
| webbrowser  | Python stdlib    |

## How It Works

1. A Chrome browser path is defined as `C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s`.
2. A tuple of URLs is defined: `stackoverflow.com`, `github.com/avinashkranjan`, `gmail.com`, `google.co.in`, `youtube.com`.
3. The `webauto()` function iterates over the URL list, prints "Opening: {url}" for each, and calls `wb.get(chrome_path).open(url)` to launch each site in Chrome.
4. The function is called immediately when the script is run.

## Project Structure

```
Chrome-Automation/
├── chrome-automation.py   # Main script
└── README.md
```

## Setup & Installation

1. Ensure Python 3.x is installed.
2. Ensure Google Chrome is installed at the path specified in the script (or modify the path).

No additional packages are required.

## How to Run

```bash
cd Chrome-Automation
python chrome-automation.py
```

Chrome will open with each of the predefined URLs in separate tabs/windows.

## Configuration

The Chrome executable path and the list of URLs are hardcoded in the script:

```python
chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
URLS = ("stackoverflow.com", "github.com/avinashkranjan", "gmail.com",
        "google.co.in", "youtube.com")
```

Modify these values directly in the script to customize.

## Testing

No formal test suite present.

## Limitations

- The Chrome executable path is hardcoded to a Windows x86 installation path (`C:/Program Files (x86)/...`). It will fail on systems where Chrome is installed elsewhere (e.g., `Program Files` on 64-bit, or Linux/macOS).
- The URL list is hardcoded — there is no command-line argument or config file to customize the sites.
- No error handling if Chrome is not installed or the path is incorrect.
- URLs do not include the `http://` or `https://` protocol prefix, relying on the browser to infer it.
- No `if __name__ == "__main__"` guard — the function runs on import as well.
