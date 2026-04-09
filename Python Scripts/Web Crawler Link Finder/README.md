# Web Crawler Link Finder

> A multi-threaded web crawler that discovers and catalogs all links within a given domain.

## Overview

This project implements a web crawler that starts from a homepage URL, discovers all links on the page, and recursively crawls each link within the same domain. It uses multiple threads for concurrent crawling and persists the queue and crawled URLs to text files.

## Features

- Multi-threaded crawling with configurable thread count (default: 8)
- Domain-restricted crawling — only follows links within the same domain
- Persistent state via text files (`queue.txt` and `crawled.txt`)
- HTML link extraction using Python's built-in `HTMLParser`
- URL normalization using `urllib.parse.urljoin`
- Automatic project directory and data file creation
- Tracks crawling progress (queue size vs. crawled count)

## Project Structure

```
Web-Crawler-Link-Finder/
├── domain.py       # Domain/subdomain extraction from URLs
├── general.py      # File I/O utilities (create dirs, read/write sets)
├── link_finder.py  # HTML parser that extracts href links
├── main.py         # Entry point with threading and job queue
├── spider.py       # Core Spider class with crawl logic
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies — uses only standard library modules:
  - `threading`, `queue` — Concurrency
  - `urllib.request`, `urllib.parse` — HTTP requests and URL parsing
  - `html.parser` — HTML link extraction
  - `os` — File system operations

## Installation

```bash
cd Web-Crawler-Link-Finder
```

No `pip install` needed — all imports are from the Python standard library.

## Usage

1. Open `main.py` and configure the target:
   ```python
   PROJECT_NAME = 'youtube'
   HOMEPAGE = 'https://www.youtube.com'
   ```
2. Optionally adjust the thread count:
   ```python
   NUMBER_OF_THREADS = 8
   ```
3. Run:
   ```bash
   python main.py
   ```

The crawler will create a project directory (e.g., `youtube/`) containing:
- `queue.txt` — URLs waiting to be crawled
- `crawled.txt` — URLs that have been crawled

## How It Works

1. **`main.py`** — Creates 8 daemon threads that pull URLs from a `Queue`. Initializes the `Spider` and starts the crawl loop.
2. **`spider.py`** — The `Spider` class manages the crawl state. On initialization, it creates the project directory and seeds the queue with the homepage. `crawl_page()` fetches a URL, extracts links, and updates the queue/crawled sets. `gather_links()` opens the URL, reads the HTML (only if `Content-Type` is `text/html`), and feeds it to `LinkFinder`.
3. **`link_finder.py`** — `LinkFinder` extends `HTMLParser` to extract all `href` attributes from `<a>` tags, resolving relative URLs with `urljoin`.
4. **`domain.py`** — Extracts the domain name from a URL using `urlparse().netloc`, keeping only the last two parts (e.g., `youtube.com`).
5. **`general.py`** — Provides file utilities: `create_project_dir()`, `create_data_files()`, `file_to_set()` (reads file lines into a set), `set_to_file()` (writes sorted set to file).

## Configuration

| Setting | File | Default |
|---|---|---|
| Project name | `main.py` | `'youtube'` |
| Homepage URL | `main.py` | `'https://www.youtube.com'` |
| Thread count | `main.py` | `8` |

## Limitations

- Uses `urllib.request.urlopen` with no timeout, headers, or user-agent — may be blocked by many websites.
- The `Spider` class uses class-level (static) variables, making it impossible to run multiple spiders simultaneously.
- Bare `except` clauses in `gather_links()` silently swallow all errors.
- No rate limiting or politeness delay between requests.
- No `robots.txt` compliance.
- No maximum crawl depth or page limit.
- The `finder` variable in `gather_links()` may be referenced before assignment if an exception occurs before `LinkFinder` is instantiated.
- Only extracts links from `<a href>` tags; does not handle JavaScript-rendered content.

## Security Notes

No sensitive data or credentials in the code.

## License

Not specified.
