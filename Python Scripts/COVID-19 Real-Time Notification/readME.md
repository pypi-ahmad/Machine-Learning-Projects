# Covid-19 Real-time Notification

## Overview

A Python script that scrapes Covid-19 data for selected Indian states and sends desktop notifications at user-defined intervals. It also translates state names into Hindi with the `english-to-hindi` library.

**Type:** Scraper / Desktop Notification Utility

## Features

- Scrapes Covid-19 statistics (total cases, active cases, deaths) from `medtalks.in`
- Filters data for user-specified Indian states
- Sends desktop push notifications using `plyer`
- Translates state names from English to Hindi using `english-to-hindi`
- Configurable notification interval (user inputs seconds at runtime)
- Supports multiple states in a single session (comma-separated input)
- Runs continuously in a loop until manually stopped

## Dependencies

Managed in `pyproject.toml` and locked in `uv.lock`:

- `Plyer` — for desktop notifications
- `requests` — for HTTP requests
- `bs4` (BeautifulSoup4) — for HTML parsing
- `english-to-hindi` — for English-to-Hindi translation

Install with:

```bash
uv sync
```

## How it works

1. The user is prompted to enter a notification interval (in seconds) and a comma-separated list of Indian state names.
2. Each state name is augmented with its Hindi translation using `eng_hindi.eth()`.
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
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Locked dependency versions
└── readME.md
```

## Setup & Installation

```bash
uv sync
```

## How to Run

```bash
uv run python Covid.py
```

When prompted:
- Enter the notification interval in seconds (e.g., `3600` for hourly).
- Enter state names separated by commas (e.g., `Maharashtra,Delhi,Karnataka`).

## Configuration

- **Notification interval**: Set at runtime via user input (in seconds).
- **States**: Set at runtime via comma-separated input.
- **Notification icon**: Loaded from `Notify_icon.ico` beside `Covid.py`.
- **Notification timeout**: Hardcoded to 5 seconds.

## Testing

No formal test suite present.

## Limitations

- The configured Medtalks data source currently returns HTTP 404, so live notifications cannot run until a replacement source is selected.
- State name matching depends on the exact format from the scraped data including Hindi transliteration — mismatches will silently skip states.
- No graceful shutdown mechanism; must be terminated manually (Ctrl+C).


