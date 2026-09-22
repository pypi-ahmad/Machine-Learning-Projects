# Chrome Automation

## Overview

A Windows CLI that previews or opens a predefined list of websites in Google Chrome. It previews URLs by default and launches Chrome only when explicitly requested.

**Type:** CLI Utility

## Features

- Previews normalized URLs before any browser action
- Opens multiple URLs as Chrome tabs with `--open`
- Accepts replacement URLs and an explicit Chrome executable path
- Uses only the Python standard library

## Dependencies

No `requirements.txt` present. Dependencies inferred from imports:

| Package     | Source           |
|-------------|------------------|
| webbrowser  | Python stdlib    |

## How It Works

1. URLs are normalized to `https://` addresses and validated.
2. Without `--open`, the script prints a preview only.
3. With `--open`, it finds Chrome in common Windows locations or uses `--chrome-path`.
4. Chrome opens each URL in a new tab.

## Project Structure

```
Chrome Automation/
├── chrome-automation.py   # Main script
└── README.md
```

## Setup & Installation

1. Install [uv](https://docs.astral.sh/uv/).
2. Run `uv sync` from this directory.
3. Install Google Chrome when you intend to use `--open`.

## How to Run

```bash
uv run python chrome-automation.py
uv run python chrome-automation.py --open
uv run python chrome-automation.py example.com https://openai.com --open
```

The first command prints a preview. Commands with `--open` launch Chrome with the selected URLs in tabs.

## Configuration

The default URL list is defined in the script. Supply URLs as positional arguments to replace it. Use `--chrome-path` when Chrome is installed outside the standard Windows locations:

```python
uv run python chrome-automation.py --chrome-path "C:\\Path\\to\\chrome.exe" --open
```

## Testing

Run `uv run python -m py_compile chrome-automation.py` to check syntax. The preview command can be run without opening a browser.

## Limitations

- The tool is designed for Windows Chrome installations.
- Opening a URL may sign in or trigger normal browser-side behavior for that site.
