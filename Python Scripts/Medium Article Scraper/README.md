# Scraping Medium Articles

> Scrapes the text content of a Medium article from its URL and saves it as a formatted text file.

## Overview

A Python script that takes a Medium article URL as input, scrapes its text content (title, headings, and body paragraphs), strips HTML tags, and saves the result as a plain text file in the `scraped_articles/` directory. Three example scraped articles are included in the project.

## Features

- Scrapes article text content from any Medium article URL
- Extracts title, section headings (as uppercase), and body text
- Strips HTML tags and converts `<br>` and `<li>` tags to newlines
- Saves output as a `.txt` file named after the article title (spaces replaced with underscores)
- Creates the `scraped_articles/` output directory if it doesn't exist
- URL validation to ensure the input is a Medium article

## Project Structure

```
Scraping Medium Articles/
├── scraping_medium.py                  # Main scraping script
├── requirements.txt                    # Python dependencies
├── scraped_articles/                   # Output directory with example scraped files
│   ├── One_month_into_the_MLH_Fellowship.txt
│   ├── One_stop_guide_to_Google_Summer_of_Code.txt
│   └── The_Pros_and_Cons_of_Open_Source_Software.txt
└── README.md
```

## Requirements

- Python 3.x
- `beautifulsoup4` (v4.9.1)
- `requests` (v2.23.0)

## Installation

```bash
cd "Scraping Medium Articles"
pip install -r requirements.txt
```

## Usage

```bash
python scraping_medium.py
```

**Prompt:**
```
Enter url of a medium article: https://medium.com/@author/article-title-abc123
```

The scraped text is saved to `scraped_articles/Article_Title.txt`.

## How It Works

1. **`get_page()`**: Prompts for a URL, validates it starts with `https://medium.com/` or `http://medium.com/` using regex, fetches the page with `requests.get()`, and parses with BeautifulSoup.
2. **`purify(text)`**: Replaces `<br>`, `<br/>`, and `<li>` tags with newlines, then strips all remaining HTML tags using regex.
3. **`collect_text(soup)`**: Extracts the article title from `<head><title>`, finds all `<h1>` elements to identify section headings, and iterates through sibling elements to collect body text. Headings are uppercased.
4. **`save_file(fin)`**: Creates the `scraped_articles/` directory if needed, constructs a filename from the article title (spaces → underscores), and writes the content to a `.txt` file with UTF-8 encoding.

## Configuration

- **Output directory**: `scraped_articles/` in the script's directory (created automatically).
- **URL validation**: Must match `https://medium.com/` prefix (also accepts `http://`).

## Limitations

- Relies on Medium's current HTML structure; Medium layout changes will break the scraper.
- The `<h1>` traversal logic using `previous_siblings` and `next_siblings` is fragile and may not work for all article formats.
- Uses a bare `except` block in the introduction extraction, silently ignoring errors.
- The `os.chdir()` call using backslash path splitting (`'\\'.join(__file__.split('/'))`) may not work correctly on all operating systems.
- No handling for Medium's paywall — paywalled articles may return incomplete content.
- No rate limiting or retry logic for failed requests.

## Security Notes

No security concerns.

## License

Not specified.
