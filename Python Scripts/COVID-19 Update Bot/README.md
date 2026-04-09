# Covid-19 Update Bot

## Overview

A Python script that fetches global Covid-19 statistics from an API and displays them as Windows 10 toast notifications at regular intervals.

**Type:** Bot / Desktop Notification Utility

## Features

- Fetches real-time Covid-19 data (confirmed cases, deaths, recovered) from a REST API
- Displays data as Windows 10 toast notifications using `win10toast`
- Runs in an infinite loop, sending a notification every 60 seconds
- Notification duration set to 20 seconds per toast

## Dependencies

- `requests` — for HTTP GET requests to the Covid-19 API
- `json` — standard library module
- `win10toast` — for Windows 10 toast notifications
- `time` — standard library module

Install with:

```bash
pip install requests win10toast
```

## How It Works

1. Sends a GET request to `https://coronavirus-19-api.herokuapp.com/all` to fetch global Covid-19 statistics.
2. Parses the JSON response to extract `cases`, `deaths`, and `recovered` fields.
3. Formats the data into a notification message string.
4. In an infinite loop:
   - Creates a `ToastNotifier` instance.
   - Shows a toast notification titled "Covid-19 Notification" with the formatted statistics.
   - Sleeps for 60 seconds before repeating.

## Project Structure

```
Covid-19-Update-Bot/
├── corona.py     # Main script
├── LICENSE       # License file
└── README.md
```

## Setup & Installation

```bash
pip install requests win10toast
```

## How to Run

```bash
python corona.py
```

The script will immediately fetch data and start showing toast notifications every 60 seconds.

## Testing

No formal test suite present.

## Limitations

- **Windows only** — `win10toast` is a Windows 10-specific notification library.
- The API endpoint (`coronavirus-19-api.herokuapp.com`) may no longer be active (Heroku free tier was discontinued).
- The data is fetched only once at startup — the notification content does not refresh on each loop iteration.
- No graceful shutdown; must be terminated manually (Ctrl+C).
- The `ToastNotifier` is re-instantiated on every loop iteration unnecessarily.
- No error handling for network failures or API response issues.





