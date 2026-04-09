# Download GeeksForGeeks Articles

> A Python script that downloads GeeksForGeeks articles as PDF files using Selenium and Chrome's built-in "Save as PDF" print functionality.

## Overview

This tool takes a GeeksForGeeks article URL as input, validates it using the `requests` library, opens it in a Selenium-controlled Chrome browser configured to print to PDF, and saves the article as a PDF file to your default download/print location.

## Features

- Downloads any GeeksForGeeks article as a PDF
- URL validation via HTTP status code check before attempting download
- Automated Chrome browser control using Selenium WebDriver
- Automatic ChromeDriver management via `webdriver-manager`
- Chrome configured with kiosk printing mode for non-interactive PDF saving
- Print preview sticky settings pre-configured for "Save as PDF"

## Project Structure

```
download GeeksForGeeks articles/
├── downloader.py      # Main script with download logic
├── requirements.txt   # Python dependencies
├── screenshot.jpg     # Screenshot of the application
└── readme.md
```

## Requirements

- Python 3.x
- Google Chrome browser installed
- `requests` 2.24.0
- `selenium` 3.141.0
- `webdriver-manager` 3.2.2

## Installation

```bash
cd "download GeeksForGeeks articles"
pip install -r requirements.txt
```

## Usage

```bash
python downloader.py
```

1. Run the script
2. When prompted, enter the full URL of a GeeksForGeeks article (e.g., `https://www.geeksforgeeks.org/what-can-i-do-with-python/`)
3. The script validates the URL, opens Chrome, and triggers the print-to-PDF action
4. The PDF is saved to Chrome's default download/print destination
5. A success message is printed upon completion

## How It Works

1. **`get_driver()`**: Configures Chrome options with `printing.print_preview_sticky_settings` set to "Save as PDF" and enables `--kiosk-printing` for automatic printing without user interaction. Uses `ChromeDriverManager().install()` to automatically download and manage the correct ChromeDriver version.
2. **`download_article(URL)`**: Opens the given URL in the configured Chrome instance and executes `window.print()` via JavaScript, which triggers the kiosk print to PDF. The browser is then closed.
3. **`__main__` block**: Prompts the user for a URL, checks if the URL returns HTTP 200 via `requests.get()`, and calls `download_article()` if valid.

## Configuration

- **Chrome print settings**: The `settings` dictionary in `get_driver()` configures the "Save as PDF" destination. Modify the `prefs` dictionary to change the PDF output path or other Chrome printing preferences.
- **ChromeDriver**: Managed automatically by `webdriver-manager`; no manual driver download required.

## Limitations

- Requires Google Chrome to be installed on the system
- The PDF is saved to Chrome's default print/download location (not configurable in the script)
- Uses `executable_path` parameter in `webdriver.Chrome()`, which is deprecated in newer Selenium versions
- Only validates the URL with a basic HTTP 200 check — does not verify it's actually a GeeksForGeeks page
- A generic `except Exception` block catches all errors during download
- No option to specify a custom output filename or directory
- The script opens a visible Chrome window (no headless mode)

## Security Notes

- The script makes HTTP requests to user-provided URLs
- No credentials are handled or stored
- `webdriver-manager` downloads ChromeDriver binaries from the internet

## License

Not specified.
