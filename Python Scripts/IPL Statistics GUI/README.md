# IPL Statistics GUI

> A Tkinter-based desktop application that scrapes and displays IPL cricket statistics from iplt20.com.

## Overview

This application provides a graphical interface for browsing Indian Premier League (IPL) statistics. Users can select a statistical category and a season, then view the table returned by the official IPL statistics page.

## Features

- GUI built with Tkinter and ttk widgets
- 12 statistical categories: Most Runs, Most Fours, Most Sixes, Most Fifties, Most Centuries, Highest Scores, Most Wickets, Most Maidens, Most Dot Balls, Best Bowling Average, Best Bowling Economy, Best Bowling Strike Rate
- Season filter supporting years 2008–2021 and an "All time" option
- Real-time web scraping from `iplt20.com/stats/`
- Displays the returned statistics table in a read-only text view
- Keeps the interface responsive while the request runs
- Shows a clear error when the service is unavailable or its page structure changes

## Project Structure

```
IPL Statistics GUI/
├── ipl.py
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python 3.13+
- `requests`
- `beautifulsoup4`
- `tkinter` (included with standard Python installations)

## Installation

```bash
cd "IPL Statistics GUI"
uv sync
```

## Usage

```bash
uv run python ipl.py
```

1. Select a statistical category from the first dropdown (defaults to "Most Runs").
2. Select a season year from the second dropdown (defaults to "All time"). The available seasons are 2008–2021, matching the original project scope.
3. Click the **Search** button to fetch and display results.
4. Results appear in the text area below, showing up to 50 player records.

## How It Works

1. **URL Generation:** `generate_url()` builds a URL like `https://www.iplt20.com/stats/{year}/{category-slug}` based on the user's dropdown selections.
2. **Web request:** The application makes a bounded HTTPS request, parses the returned HTML with Beautiful Soup, and looks for the official statistics table.
3. **Display:** The extracted table text is rendered in a read-only Tkinter text widget. The request runs in a background thread so the window remains usable.

## Configuration

- **Category-to-slug mapping:** Defined in the `CATEGORIES` dictionary at the top of the script.
- **Year options:** Defined in `SEASONS` (2008–2021 plus All time).
- **Request timeout:** 20 seconds.
- **Record limit:** The application displays the full table returned by the page.

## Limitations

- Depends on the structure and availability of iplt20.com. A page-layout change can prevent table extraction; the app will report that error instead of displaying misleading output.
- Year options remain limited to the project's original 2008–2021 scope and do not automatically include new IPL seasons.
- This app reads public statistics pages only; it does not authenticate or modify IPL data.

## Security Notes

No security concerns identified.

## License

Not specified.
