# Live Cricket Score

> A Python script that scrapes live cricket scores from Cricbuzz and displays them as Windows desktop notifications.

## Overview

This script continuously scrapes the Cricbuzz live scores page, extracts match headers and scores, and sends Windows 10 toast desktop notifications for each active match. It runs in an infinite loop, checking for updates and sending notifications with a 10-second delay between each match.

## Features

- Scrapes live cricket match data from Cricbuzz (`cricbuzz.com/cricket-match/live-scores`)
- Extracts match headers (team names/match info) and live scores
- Sends Windows 10 desktop toast notifications via `win10toast`
- Uses a custom icon (`ipl.ico`) for notifications
- Notifications persist for 30 seconds each
- Runs continuously in an infinite loop

## Project Structure

```
Live-Cricket-Score/
├── ipl.ico
├── live_score.py
└── requirements.txt
```

## Requirements

- Python 3.x
- Windows 10 (required for toast notifications)
- `beautifulsoup4==4.9.3`
- `bs4==0.0.1`
- `win10toast==0.9`
- `pypiwin32==223`
- `pywin32==228`
- `urllib3==1.26.5`
- `soupsieve==2.0.1`

## Installation

```bash
cd "Live-Cricket-Score"
pip install -r requirements.txt
```

## Usage

```bash
python live_score.py
```

The script will start scraping Cricbuzz and display desktop notifications for each live match. Press `Ctrl+C` to stop.

## How It Works

1. **HTTP Request:** Sends a GET request to `http://www.cricbuzz.com/cricket-match/live-scores` using `urllib.request.urlopen` with a custom User-Agent header (`XYZ/3.0`) and a 20-second timeout.
2. **HTML Parsing:** Parses the response with BeautifulSoup using the `html.parser` backend.
3. **Data Extraction:** Finds all `<div>` elements with class `cb-col cb-col-100 cb-plyr-tbody cb-rank-hdr cb-lv-main`, then extracts:
   - **Header:** From the child `<div>` with class `cb-col-100 cb-col cb-schdl` (match/team info).
   - **Score:** From the child `<div>` with class `cb-scr-wll-chvrn cb-lv-scrs-col` (live score text).
4. **Notification:** Calls `notify(header, score)` which uses `ToastNotifier().show_toast()` to display a Windows toast notification titled "CRICKET LIVE SCORE" with the score as the body, displayed for 30 seconds with the `ipl.ico` icon.
5. **Loop:** After notifying for each match (with a 10-second `time.sleep()` between), the outer `while True` loop repeats the entire scrape cycle.

## Configuration

- **Target URL:** Hardcoded as `http://www.cricbuzz.com/cricket-match/live-scores`.
- **User-Agent:** Set to `XYZ/3.0` in the request header.
- **Request timeout:** 20 seconds.
- **Notification duration:** 30 seconds per notification.
- **Delay between notifications:** 10 seconds via `time.sleep(10)`.
- **Notification icon:** `ipl.ico` (must be in the same directory).

## Limitations

- **Windows only:** Depends on `win10toast` which only works on Windows 10.
- No error handling for network failures, missing HTML elements, or Cricbuzz structure changes.
- The infinite loop with `time.sleep(10)` between each match notification means the script can flood the desktop with notifications if many matches are live.
- Uses `http://` instead of `https://` for the Cricbuzz URL.
- The `notify()` function parameter `title` is accepted but not used — the notification title is hardcoded as "CRICKET LIVE SCORE".
- If Cricbuzz changes its HTML class names, the scraper will break silently.
- No graceful exit mechanism other than `Ctrl+C`.

## Security Notes

No security concerns identified.

## License

Not specified.
