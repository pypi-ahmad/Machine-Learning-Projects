# Download Images from Website

> A Python script that scrapes and downloads all images from a given website using Selenium and BeautifulSoup.

## Overview

This tool opens a website in a Selenium-controlled Chrome browser, extracts the page's HTML, parses all `<img>` tags using BeautifulSoup, and downloads each image to a local `output` folder. It supports JPEG, JPG, PNG, and GIF formats.

## Features

- Automated web page loading via Selenium WebDriver
- HTML parsing with BeautifulSoup to extract all `<img>` tags with `src` attributes
- Downloads images in JPEG, JPG, PNG, and GIF formats
- Automatic file extension detection from image URL
- Images saved to a local `output` directory (created automatically if it doesn't exist)
- Sequential file naming (1.jpg, 2.png, etc.)

## Project Structure

```
Download_images_from_website/
├── scrap-img.py       # Main scraping and download script
├── requirements.txt   # Python dependencies
└── Readme.md
```

## Requirements

- Python 3.x
- Google Chrome browser installed
- ChromeDriver executable (must be downloaded separately and path provided at runtime)
- `selenium` 3.141.0
- `requests`
- `beautifulsoup4`
- `lxml`

> **Note:** The `requirements.txt` only lists `selenium==3.141.0`. You will also need `requests`, `beautifulsoup4`, and `lxml` installed.

## Installation

```bash
cd "Download_images_from_website"
pip install -r requirements.txt
pip install requests beautifulsoup4 lxml
```

## Usage

```bash
python scrap-img.py
```

1. When prompted, enter the full path to your ChromeDriver executable (e.g., `E:\web scraping\chromedriver_win32\chromedriver.exe`)
2. Enter the URL of the website you want to scrape images from
3. The script loads the page in Chrome, waits 60 seconds for content to load, then downloads all images to the `output` folder
4. Progress is printed to the console as each image is downloaded

## How It Works

1. **`get_url(path, url)`**: Launches a Chrome browser using the provided ChromeDriver path, navigates to the URL, and retrieves the full rendered HTML via `document.documentElement.outerHTML` JavaScript execution.
2. **`get_img_links(res)`**: Parses the HTML with BeautifulSoup using the `lxml` parser and finds all `<img>` tags that have a `src` attribute.
3. **`download_img(img_link, index)`**: Downloads a single image from the given URL using `requests.get()`. Detects the file extension by searching the URL for `.jpeg`, `.jpg`, `.png`, or `.gif` (defaults to `.jpg`). Saves the image to the `output` directory with a sequential numeric filename.
4. **Main flow**: Calls the functions in sequence, waits 60 seconds with `time.sleep(60)` after page load, creates the `output` directory if it doesn't exist, and iterates through all found image links.

## Configuration

- **ChromeDriver path**: Provided interactively at runtime via `input()`
- **Output directory**: Hardcoded as `"output"` in the script — modify the `output` variable to change
- **Wait time**: Hardcoded `time.sleep(60)` — adjust for faster or slower-loading pages
- **Supported extensions**: `.jpeg`, `.jpg`, `.png`, `.gif` — add more to the `extensions` list in `download_img()` if needed

## Limitations

- Requires manual ChromeDriver download and path entry (does not use `webdriver-manager`)
- The 60-second wait (`time.sleep(60)`) is hardcoded and may be too long or too short depending on the website
- Only detects images in `<img>` tags with `src` attributes — misses CSS background images, lazy-loaded images, or images in `<picture>` elements
- Extension detection is based on string matching in the URL, which may fail for URLs without visible extensions
- The `except Exception: pass` silently swallows all download errors
- Uses `f.close()` explicitly after a `with` block (redundant)
- Backslash path separator (`\\`) in `download_img()` makes it Windows-specific
- The Chrome browser window remains open after scraping

## Security Notes

- The script downloads content from user-provided URLs to local disk
- No input validation on the ChromeDriver path or URL
- Image data is written directly to disk without sanitization

## License

Not specified.
