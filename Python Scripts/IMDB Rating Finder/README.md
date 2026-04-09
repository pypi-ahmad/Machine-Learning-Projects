# Find IMDb Rating

> A Python web scraper that fetches IMDb ratings and genres for movies in a local directory and exports the results to a CSV file.

## Overview

This script scans a local directory for film files, searches IMDb for each film by name, scrapes the rating and genre from the search results page, and saves the data to a CSV file (`film_ratings.csv`) using Pandas.

## Features

- Scans a local directory and extracts film names from filenames (without extensions)
- Searches IMDb for each film using the title search endpoint
- Scrapes film ratings and genres from IMDb search results using BeautifulSoup
- Exports results (film name, rating, genre) to a CSV file
- Uses a persistent `requests.Session` for efficient HTTP connections

## Project Structure

```
Find_imdb_rating/
├── find_IMDb_rating.py   # Main scraping script
├── requirements.txt      # Python dependencies
└── .gitignore            # Git ignore file
```

## Requirements

- Python 3.x
- `beautifulsoup4==4.9.1`
- `requests==2.24.0`
- `pandas==1.1.2`

Full dependency list in `requirements.txt`:
```
beautifulsoup4==4.9.1
certifi==2020.6.20
chardet==3.0.4
idna==2.10
pandas==1.1.2
python-dateutil==2.8.1
pytz==2020.1
requests==2.24.0
six==1.15.0
soupsieve==2.0.1
urllib3==1.25.10
```

## Installation

```bash
cd "Find_imdb_rating"
pip install -r requirements.txt
```

## Usage

```bash
python find_IMDb_rating.py
```

When prompted, enter the path to your films directory:

```
Enter the path where your films are: /Users/you/Movies
```

The script will:
1. List all files in the given directory
2. Search IMDb for each film
3. Print the search URLs to the console
4. Generate `film_ratings.csv` in the current directory

## How It Works

1. Lists all files in the user-specified directory using `os.listdir()`.
2. Strips file extensions to get film titles.
3. For each title, constructs an IMDb search URL: `https://www.imdb.com/search/title/?title={query}`.
4. Fetches the search results page and parses it with BeautifulSoup.
5. Iterates through `div.lister-item-content` containers to find matching titles.
6. Extracts the `data-value` attribute from the ratings div and the genre span.
7. Collects all results into a Pandas DataFrame and exports to CSV.

## Configuration

No configuration files. The film directory is provided interactively at runtime.

## Limitations

- Relies on IMDb's HTML structure (CSS classes like `lister-item-content`, `ratings-imdb-rating`) — will break if IMDb redesigns their search page
- Film title matching is case-insensitive but exact substring match — may return wrong results for common words
- Only picks the first matching result from each search page
- Films that don't match any IMDb result are silently skipped
- Generic `except Exception` catches all errors without useful error messages
- No rate limiting — may be blocked by IMDb for rapid successive requests
- The dependency versions in `requirements.txt` are outdated

## Security Notes

No sensitive credentials in the code. IMDb scraping may violate their Terms of Service.

## License

Not specified.
