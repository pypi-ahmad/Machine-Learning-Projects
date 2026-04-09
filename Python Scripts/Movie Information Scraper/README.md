# Movie Information Scraper

> A Python script that scrapes detailed movie information from IMDB by searching for a movie title and parsing the result pages.

## Overview

This command-line tool prompts the user for a movie name, searches IMDB's title search, navigates to the first matching feature film page, and extracts comprehensive details including rating, runtime, genres, directors, writers, cast, and plot summary.

## Features

- Searches IMDB title search filtered to feature films
- Extracts movie name, year, IMDB rating, and runtime
- Retrieves genres, release date, directors, writers, and cast
- Fetches the full plot summary from a separate IMDB plot page
- Handles missing data gracefully (e.g., unrated films, unavailable runtime)
- Returns results as a structured dictionary

## Project Structure

```
Movie Information Scraper/
├── movieInfoScraper.py    # Main scraper script
├── requirements.txt       # Python dependencies
└── Screenshot.png         # Sample output screenshot
```

## Requirements

- Python 3.x
- beautifulsoup4
- requests == 2.23.0

## Installation

```bash
cd "Movie Information Scraper"
pip install -r requirements.txt
```

## Usage

```bash
python movieInfoScraper.py
```

When prompted, enter a movie name:

```
Enter the movie name whose details are to be fetched
The Dark Knight

The Dark Knight (2008)
Rating: 9.0
Runtime: 2h 32min
Release Date: 18 July 2008 (USA)
Genres: Action, Crime, Drama
Director: Christopher Nolan
Writer: Jonathan Nolan, Christopher Nolan
Cast: Christian Bale, Heath Ledger, Aaron Eckhart
Plot Summary:
 When the menace known as the Joker wreaks havoc...
```

## How It Works

1. **Search**: Constructs a search URL (`/search/title?title=...&title_type=feature`) and fetches the results page
2. **First result**: Finds the first `<h3 class="lister-item-header">` element to get the movie link
3. **Movie page**: Fetches the individual movie page and parses:
   - Year from `<span id="titleYear">`
   - Rating from `<div class="ratingValue">`
   - Runtime from `<time>` in subtext
   - Genres from anchor tags in subtext
   - Release date from the "See more release dates" link
   - Directors, writers, cast from `credit_summary_item` divs
4. **Plot**: Makes an additional request to the `/plotsummary` page and extracts the first plot summary

The `getMovieDetails()` function returns a dictionary, and the `__main__` block handles user I/O and display.

## Configuration

No configuration files. All URLs are hardcoded to IMDB's domain (`https://www.imdb.com`).

## Limitations

- Relies on IMDB's specific HTML structure — will break if IMDB changes their page layout (class names like `lister-item-header`, `ratingValue`, `credit_summary_item`)
- Only returns the first search result, not allowing the user to choose
- No error handling for network failures or connection timeouts
- When writer details are missing, the code silently assigns the cast list to writers and vice versa (fallback logic in the `except IndexError` block)
- No rate limiting or request delays, which may trigger IMDB blocking
- Hardcoded to search only feature films (`title_type=feature`)

## License

Not specified.
