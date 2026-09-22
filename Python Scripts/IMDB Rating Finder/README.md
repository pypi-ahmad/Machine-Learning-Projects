# Find IMDb Rating

> A Python web scraper that fetches IMDb ratings and genres for films in a local directory and exports the results to a CSV file.

## Overview

This script scans a local directory for film files, searches IMDb by title, extracts ratings and genres from the results, and saves the data to `film_ratings.csv` with Pandas.

## Features

- Scans a local directory and extracts film names from filenames (without extensions)
- Searches IMDb for each film through the title search endpoint
- Scrapes film ratings and genres from IMDb search results using BeautifulSoup
- Exports results (film name, rating, genre) to a CSV file
- Uses a persistent `requests.Session` for efficient HTTP connections

## Project Structure

```
Find_imdb_rating/
├── find_IMDb_rating.py   # Main scraping script
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Locked dependency versions
└── .gitignore            # Git ignore file
```

## Requirements

- Python 3.13+
- `beautifulsoup4`, `requests`, and `pandas`, managed by uv in `pyproject.toml`

## Installation

```bash
cd "Find_imdb_rating"
uv sync
```

## Usage

```bash
uv run python find_IMDb_rating.py path/to/films
```

Specify a new output file or request delay when needed:

```bash
uv run python find_IMDb_rating.py path/to/films --output ratings.csv --delay 1
```

The script will:
1. List all files in the given directory
2. Search IMDb for each film
3. Reports unavailable results and request failures without stopping the run
4. Generates `film_ratings.csv` in the current directory by default

## How it works

1. Lists files in the supplied directory and derives titles from their filename stems.
2. Searches IMDb with request parameters, a timeout, and a descriptive user agent.
3. Parses matching legacy result containers for title, rating, and genre.
4. Delays between requests and exports collected results to a new CSV file.

## Configuration

No configuration files. Provide the film directory as an argument; use `--output` and `--delay` to control output and pacing.

## Limitations

- Relies on IMDb's HTML structure (CSS classes like `lister-item-content`, `ratings-imdb-rating`) — will break if IMDb redesigns their search page
- Film title matching is case-insensitive but exact substring match — may return wrong results for common words
- Only picks the first matching result from each search page
- IMDb can change its markup or restrict automated requests, which may produce unavailable results.
- Matching is title-based and can select an unintended result for ambiguous filenames.
- Existing output CSV files are not overwritten.

## Security Notes

No sensitive credentials in the code. IMDb scraping may violate their Terms of Service.

## License

Not specified.
