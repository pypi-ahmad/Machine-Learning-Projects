# Web Scraping YouTube Comments

> A Selenium-based script that scrapes usernames and comments from a YouTube video and saves them to a CSV file.

## Overview

This script uses Selenium WebDriver to open a YouTube video in Chrome, scrolls down to load comments, extracts all visible comment authors and text, and exports the data to a CSV file.

## Features

- Automated YouTube video loading via Selenium ChromeDriver
- Page scrolling to trigger comment loading
- Extracts comment author names and comment text
- Exports results to a CSV file with `Author` and `Comment` columns
- UTF-8 encoding support for international characters

## Project Structure

```
Web_scraping_a_youtube_comment/
├── demo.gif
├── README.md
├── requirements.txt
└── webscrapindcomment.py
```

## Requirements

- Python 3.x
- `selenium` (3.141.0) — Browser automation
- Google Chrome browser
- [ChromeDriver](https://chromedriver.chromium.org/downloads) matching your Chrome version

## Installation

```bash
cd Web_scraping_a_youtube_comment
pip install -r requirements.txt
```

Download ChromeDriver and place it in an accessible location.

## Usage

1. Open `webscrapindcomment.py` and update the ChromeDriver path:
   ```python
   driver = webdriver.Chrome(r"C:/path/to/chromedriver.exe")
   ```
2. Update the YouTube video URL:
   ```python
   driver.get('https://www.youtube.com/watch?v=YOUR_VIDEO_ID')
   ```
3. Update the CSV output path:
   ```python
   filename = 'C:/path/to/commentlist.csv'
   ```
4. Run:
   ```bash
   python webscrapindcomment.py
   ```

## How It Works

1. Opens Chrome via Selenium WebDriver with the specified ChromeDriver path.
2. Navigates to the hardcoded YouTube video URL.
3. Scrolls the page twice (to 500px, then to 3000px after a 5-second wait) to trigger YouTube's lazy-loaded comments.
4. Finds all comment author elements using XPath: `//*[@id="author-text"]`.
5. Finds all comment text elements using XPath: `//*[@id="content-text"]`.
6. Zips authors and comments together into a list of dictionaries.
7. Writes the data to a CSV file using `csv.DictWriter`.

## Configuration

| Setting | Location | Value |
|---|---|---|
| ChromeDriver path | `webscrapindcomment.py` | `r"C:/Users/hp/Anaconda3/chromedriver.exe"` |
| YouTube video URL | `webscrapindcomment.py` | `'https://www.youtube.com/watch?v=iFPMz36std4'` |
| CSV output path | `webscrapindcomment.py` | `'C:/Users/hp/Desktop/commentlist.csv'` |

## Limitations

- All paths (ChromeDriver, video URL, output CSV) are hardcoded — no command-line arguments.
- Only scrapes comments visible after two scroll actions; does not load all comments on videos with many comments.
- The 5-second `time.sleep()` delay is fixed and may be insufficient on slow connections or excessive on fast ones.
- Uses deprecated Selenium methods (`find_elements_by_xpath`) — newer Selenium versions require `find_element(By.XPATH, ...)`.
- YouTube frequently changes its DOM structure, which may break the XPath selectors.
- The browser window stays open after scraping (no `driver.quit()`).
- No error handling for network issues, missing elements, or ChromeDriver version mismatches.

## Security Notes

No sensitive data or credentials in the code.

## License

Not specified.
