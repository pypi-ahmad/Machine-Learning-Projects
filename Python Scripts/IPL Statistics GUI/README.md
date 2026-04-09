# IPL Statistics GUI

> A Tkinter-based desktop application that scrapes and displays IPL cricket statistics from iplt20.com.

## Overview

This application provides a graphical interface for browsing Indian Premier League (IPL) statistics. Users can select a statistical category (e.g., Most Runs, Most Wickets) and a season year, then view the top players in a formatted table. Data is scraped in real-time from the official IPL website.

## Features

- GUI built with Tkinter and ttk widgets
- 12 statistical categories: Most Runs, Most Fours, Most Sixes, Most Fifties, Most Centuries, Highest Scores, Most Wickets, Most Maidens, Most Dot Balls, Best Bowling Average, Best Bowling Economy, Best Bowling Strike Rate
- Season filter supporting years 2008–2021 and an "All time" option
- Real-time web scraping from `iplt20.com/stats/`
- Displays up to 50 player records in a formatted text widget

## Project Structure

```
IPL Statistics GUI/
├── ipl.py
└── requirements.txt
```

## Requirements

- Python 3.x
- `requests`
- `beautifulsoup4`
- `tkinter` (included with standard Python installations)

## Installation

```bash
cd "IPL Statistics GUI"
pip install -r requirements.txt
```

## Usage

```bash
python ipl.py
```

1. Select a statistical category from the first dropdown (defaults to "Most Runs").
2. Select a season year from the second dropdown (defaults to "All time").
3. Click the **Search** button to fetch and display results.
4. Results appear in the text area below, showing up to 50 player records.

## How It Works

1. **URL Generation:** `generate_url()` builds a URL like `https://www.iplt20.com/stats/{year}/{category-slug}` based on the user's dropdown selections.
2. **Web Scraping:** `scrape_results()` sends a GET request to the generated URL, parses the HTML with BeautifulSoup, and extracts the table with class `table table--scroll-on-tablet top-players`.
3. **Data Formatting:** Table rows are extracted and formatted with fixed-width columns (20 characters each), truncating longer values.
4. **Display:** The formatted text is inserted into a read-only `tk.Text` widget.

## Configuration

- **Category-to-slug mapping:** Defined in the `categories` dictionary at the top of the script.
- **Year options:** Hardcoded in the `year['values']` tuple (2008–2021).
- **Window size:** Set to `800x850` pixels.
- **Record limit:** Displays up to 50 records (`table_data[:51]` including the header row).

## Limitations

- Depends on the structure of iplt20.com — if the website changes its HTML layout, scraping will break.
- No error handling for network failures or missing data on the page.
- Year options are hardcoded and do not automatically include new IPL seasons.
- Column values are truncated to 20 characters, which may cut off longer player names or descriptions.
- The text display area uses a fixed-width format that may not render well for all screen sizes.

## Security Notes

No security concerns identified.

## License

Not specified.
