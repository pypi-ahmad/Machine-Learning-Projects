# Download GeeksForGeeks Articles

> A Python script that downloads GeeksForGeeks articles as PDF files using Selenium and Chrome's built-in "Save as PDF" print functionality.

## Overview

This tool validates a GeeksForGeeks article URL, renders it in headless Chrome, and saves the page as a PDF at the chosen path.

## Features

- Downloads public GeeksForGeeks articles as PDF
- Validates the GeeksForGeeks host before opening Chrome
- Uses Selenium Manager for compatible ChromeDriver handling
- Saves directly through Chrome's PDF command without a print dialog
- Supports a chosen output path

## Project Structure

```
download GeeksForGeeks articles/
├── downloader.py      # Main script with download logic
├── pyproject.toml     # Project metadata and dependencies
├── uv.lock            # Locked dependency versions
├── screenshot.jpg     # Screenshot of the application
└── readme.md
```

## Requirements

- Python 3.13+
- Google Chrome browser installed
- `requests` and `selenium`, managed by uv in `pyproject.toml`

## Installation

```bash
cd "download GeeksForGeeks articles"
uv sync
```

## Usage

```bash
uv run python downloader.py
```

1. Run the script
2. When prompted, enter the full URL of a GeeksForGeeks article (e.g., `https://www.geeksforgeeks.org/what-can-i-do-with-python/`)
3. The script validates the URL, opens headless Chrome, and saves `article.pdf` in the current directory
4. A success message is printed upon completion

Pass URL and output path directly when preferred:

```bash
uv run python downloader.py https://www.geeksforgeeks.org/what-can-i-do-with-python/ --output output/article.pdf
```

## How It Works

1. **`validate_article_url()`**: Checks for an HTTP(S) GeeksForGeeks URL.
2. **`download_article()`**: Verifies the URL is reachable, opens it in headless Chrome, and calls Chrome's `Page.printToPDF` command.
3. **`main()`**: Handles interactive or direct URL input and refuses to overwrite an existing PDF.

## Configuration

- **Output file**: Use `--output` to choose the PDF destination. The default is `article.pdf` in the current directory.
- **ChromeDriver**: Managed automatically by Selenium Manager; no manual driver download is required.

## Limitations

- Requires Google Chrome to be installed on the system
- Requires Chrome and a compatible driver managed by Selenium Manager
- Page layout and content depend on the current GeeksForGeeks website
- Only public pages that render in Chrome can be saved

## Security Notes

- The script makes HTTP requests only to GeeksForGeeks URLs
- No credentials are handled or stored
- Selenium Manager may download a ChromeDriver binary from the internet

## License

Not specified.
