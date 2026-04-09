# Wikipedia Scrapper in Python

> A script showcasing the `wikipedia` Python library's capabilities for searching, summarizing, and extracting content from Wikipedia.

## Overview

This script uses various features of the `wikipedia` library: searching for articles, getting search suggestions, fetching article summaries (in English and French), and accessing page metadata including title, URL, full content, images, and reference links.

## Features

- Search Wikipedia for articles matching a query (`wikipedia.search()`)
- Get search suggestions for partial queries (`wikipedia.suggest()`)
- Fetch article summaries (`wikipedia.summary()`)
- Switch language for summaries (demonstrates English and French)
- Access full page content, title, URL, images, and reference links via `wikipedia.page()`

## Project Structure

```
Wikipedia Scrapper in Python/
├── README.md
└── Wikipedia Scrapper in Python.py
```

## Requirements

- Python 3.x
- `wikipedia` — Python wrapper for the Wikipedia API

## Installation

```bash
cd "Wikipedia Scrapper in Python"
pip install wikipedia
```

## Usage

```bash
python "Wikipedia Scrapper in Python.py"
```

The script runs a series of demonstrations using "Python" as the topic:

1. Searches for "Python" and prints matching article titles
2. Gets a suggestion for the partial query "Pyth"
3. Prints the English summary for "Python"
4. Switches to French (`wiki.set_lang("fr")`) and prints the French summary
5. Switches back to English and fetches the full "Python" page
6. Prints the page title, URL, full content, all image URLs, and all reference links

## How It Works

1. `wiki.search("Python")` — Returns a list of article titles matching the query.
2. `wiki.suggest("Pyth")` — Returns Wikipedia's autocomplete suggestion.
3. `wiki.summary("Python")` — Fetches the summary section of the article.
4. `wiki.set_lang("fr")` — Changes the Wikipedia language for subsequent calls.
5. `wiki.page("Python")` — Returns a `WikipediaPage` object with properties: `.title`, `.url`, `.content`, `.images`, `.links`.

## Configuration

- The search topic `"Python"` is hardcoded throughout the script.
- Language switching is hardcoded between `"fr"` (French) and `"en"` (English).

## Limitations

- All queries are hardcoded; no user input or command-line arguments.
- Prints large amounts of output (full article content, all images, all links) directly to stdout.
- No error handling for disambiguation pages, missing articles, or network issues.
- The `wikipedia` library may raise `DisambiguationError` for ambiguous queries like "Python".
- Sets language globally (`set_lang`), which could cause issues if extended to handle multiple languages simultaneously.

## Security Notes

No sensitive data or credentials in the code.

## License

Not specified.
