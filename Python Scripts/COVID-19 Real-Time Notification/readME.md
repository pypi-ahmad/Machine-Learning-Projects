# Covid-19 Real-time Notification

## Overview

A Python script that scrapes real-time Covid-19 data for specific Indian states from a web source and delivers desktop notifications at user-defined intervals. Includes Hindi translation of state names using the `englisttohindi` library.

**Type:** Scraper / Desktop Notification Utility

## Features

- Scrapes Covid-19 statistics (total cases, active cases, deaths) from `medtalks.in`
- Filters data for user-specified Indian states
- Sends desktop push notifications using `plyer`
- Translates state names from English to Hindi using `englisttohindi`
- Configurable notification interval (user inputs seconds at runtime)
- Supports multiple states in a single session (comma-separated input)
- Runs continuously in a loop until manually stopped

## Dependencies

Listed in `requirements.txt`:

- `Plyer` — for desktop notifications
- `requests` — for HTTP requests
- `bs4` (BeautifulSoup4) — for HTML parsing
- `englisttohindi` — for English-to-Hindi transliteration

Install with:

```bash
pip install plyer requests beautifulsoup4 englisttohindi
```

## How It Works

1. The user is prompted to enter a notification interval (in seconds) and a comma-separated list of Indian state names.
2. Each state name is augmented with its Hindi transliteration using `EngtoHindi`.
3. In an infinite loop:
   - The script sends a GET request to `https://www.medtalks.in/live-corona-counter-india`.
   - The HTML response is parsed with BeautifulSoup, extracting data from the `<tbody>` table rows.
   - The table data is split into per-state records.
   - For each matching state, a desktop notification is sent via `plyer.notification.notify()` showing total, active, and death counts.
   - A 2-second delay is added between notifications for different states.
   - The loop sleeps for the user-specified interval before repeating.

## Project Structure

```
Covid-19_Real-time_Notification/
├── Covid.py            # Main script
├── Notify_icon.ico     # Icon file used for desktop notifications
├── requirements.txt    # Dependencies
└── readME.md
```

## Setup & Installation

```bash
pip install plyer requests beautifulsoup4 englisttohindi
```

## How to Run

```bash
python Covid.py
```

When prompted:
- Enter the notification interval in seconds (e.g., `3600` for hourly).
- Enter state names separated by commas (e.g., `Maharashtra,Delhi,Karnataka`).

## Configuration

- **Notification interval**: Set at runtime via user input (in seconds).
- **States**: Set at runtime via comma-separated input.
- **Notification icon**: Hardcoded path `./Covid-19_Real-time_Notification/Notify_icon.ico` — may need adjustment depending on working directory.
- **Notification timeout**: Hardcoded to 5 seconds.

## Testing

No formal test suite present.

## Limitations

- The data source URL (`medtalks.in`) may no longer be active or may have changed its HTML structure since the script was written.
- The notification icon path is hardcoded with a relative path that assumes a specific working directory.
- State name matching depends on the exact format from the scraped data including Hindi transliteration — mismatches will silently skip states.
- No graceful shutdown mechanism; must be terminated manually (Ctrl+C).


