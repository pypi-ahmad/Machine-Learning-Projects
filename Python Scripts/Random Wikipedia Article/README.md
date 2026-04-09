# Random Wikipedia Article

> A Python script that fetches a random Wikipedia article and saves its content to a text file.

## Overview

This script uses the `Special:Random` Wikipedia URL to fetch a random article, parses the HTML with BeautifulSoup, and saves the article's heading and paragraph text to a local file called `random_wiki.txt`.

## Features

- Fetches a random Wikipedia article using `Special:Random`
- Extracts the article heading (`<h1>` tag)
- Extracts all paragraph text (`<p>` tags)
- Saves the content to `random_wiki.txt` with UTF-8 encoding
- Raises an HTTP error if the request fails (`raise_for_status()`)

## Project Structure

```
Random_Wikipedia_Article/
├── wiki_random.py       # Main script
├── requirements.txt     # Dependencies
└── README.md
```

## Requirements

- Python 3.x
- `requests`
- `beautifulsoup4`

The `requirements.txt` lists `HTMLParser==0.0.2`, but the code actually uses `beautifulsoup4` and `requests` (not the `HTMLParser` package).

## Installation

```bash
cd "Random_Wikipedia_Article"
pip install requests beautifulsoup4
```

## Usage

```bash
python wiki_random.py
```

Output:

```
File Saved as random_wiki.txt
```

The file `random_wiki.txt` will be created in the current directory containing the article's heading and paragraph text.

## How It Works

1. Sends a GET request to `https://en.wikipedia.org/wiki/Special:Random`, which redirects to a random article.
2. Calls `res.raise_for_status()` to ensure the request succeeded.
3. Parses the response HTML with `BeautifulSoup(res.text, "html.parser")`.
4. Opens (or creates) `random_wiki.txt` in write mode with UTF-8 encoding.
5. Writes the `<h1>` heading text as the first line.
6. Iterates over all `<p>` tags and appends their text to the file.

## Configuration

- Output filename is hardcoded as `"random_wiki.txt"`
- The parser used is `"html.parser"` (Python's built-in HTML parser)

## Limitations

- Overwrites `random_wiki.txt` on each run (no append or unique naming)
- Only extracts `<h1>` and `<p>` tags — images, tables, lists, and other content are ignored
- No command-line arguments for specifying output file or article topic
- The `requirements.txt` lists `HTMLParser==0.0.2` which doesn't match the actual imports (`beautifulsoup4`, `requests`)
- No error handling for network timeouts or missing elements

## Security Notes

No security concerns identified.

## License

Not specified.
