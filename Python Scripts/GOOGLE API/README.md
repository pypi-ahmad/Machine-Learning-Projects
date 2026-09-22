# Google API — Geocoding Visualizer

> Geocode addresses using the Google Maps API (or a fallback service) and visualize locations on an interactive map.

## Overview

A two-script pipeline that geocodes addresses from a data file into a SQLite database, then exports the results to a JavaScript file for visualization on a Google Maps-based HTML page. This is a project based on the "Python for Everybody" (py4e) course.

## Features

- Reads addresses from a `where.data` text file (one address per line)
- Geocodes addresses through the Google Maps Geocoding API
- Caches results in a SQLite database (`geodata.sqlite`) to avoid redundant API calls
- Rate-limits API requests (pauses every 10 requests)
- Exports geocoded coordinates and formatted addresses to `where.js`
- Visualizes locations on an interactive map via `where.html`

## Project Structure

```
GOOGLE API/
├── geoload.py       # Geocodes addresses and stores results in SQLite
├── geodump.py       # Reads SQLite data and writes to where.js
├── where.html       # HTML page that renders locations on a map
├── where.js         # Auto-generated JavaScript data file (output of geodump.py)
├── Capture1.PNG     # Screenshot
├── Capture2.PNG     # Screenshot
├── Capture3.PNG     # Screenshot
└── README.md
```

## Requirements

- Python 3.13 or later
- No external packages (uses only standard library: `urllib`, `sqlite3`, `json`, `codecs`, `ssl`, `http`, `time`)
- A `where.data` file containing addresses to geocode (one per line)

## Installation

```bash
cd "Python Scripts/GOOGLE API"
uv sync --no-config
```

No package installation is needed. Set `GOOGLE_API_KEY` in the Windows user environment, relaunch the coding host if it was recently added, then create a `where.data` file with addresses:
```
Ann Arbor, MI
Tokyo, Japan
Paris, France
```

## Usage

### Step 1: Geocode addresses

```bash
uv run --no-config python geoload.py
```

This reads `where.data`, geocodes each address (up to 200 per run), and stores results in `geodata.sqlite`. Run multiple times to process more than 200 addresses.

### Step 2: Export to JavaScript

```bash
uv run --no-config python geodump.py
```

This reads `geodata.sqlite` and writes coordinate data to `where.js`.

### Step 3: View the map

Open `where.html` in a web browser to see all geocoded locations plotted on a map.

## How It Works

### `geoload.py`
1. Creates a SQLite table `Locations` with columns `address` (TEXT) and `geodata` (TEXT).
2. Reads `where.data` line by line. For each address:
   - Checks if already cached in the database (skips if found).
   - Constructs a URL with the address as a parameter and optional API key.
   - Fetches JSON from the geocoding service.
   - Stores the raw JSON response in the database.
3. Limits to 200 addresses per run with a 5-second pause every 10 requests.

### `geodump.py`
1. Reads all rows from the `Locations` table.
2. Parses each JSON response, extracts `lat`, `lng`, and `formatted_address`.
3. Writes entries as JavaScript array elements to `where.js` in the format: `[lat, lng, 'address']`.

## Configuration

- **API Key:** Set `GOOGLE_API_KEY` in the process environment. Never add it to source code or a committed `.env` file.
- **Rate Limit:** Hardcoded 200-address limit per run and 5-second pause every 10 requests.
- **SSL Verification:** Disabled (`ctx.check_hostname = False`, `ctx.verify_mode = ssl.CERT_NONE`).

## Limitations

- **No `where.data` file included** — must be created manually.
- **200-address cap per run** — must restart the script to continue processing.
- **Bare `except` blocks** — errors during JSON parsing and database lookups are silently ignored.
- **`memoryview` usage** for database storage may cause compatibility issues with some SQLite versions.
- Google API access, quotas, billing, and response availability are managed by the account that owns `GOOGLE_API_KEY`.

## Security Notes

- TLS certificate verification remains enabled.
- The loader reads `GOOGLE_API_KEY` only at runtime and does not print or store it.

## License

Not specified.
