# Take-A-Break

> A collection of Python scripts that open a YouTube video in the browser after a timed delay, encouraging the user to take a break.

## Overview

This project contains three small scripts exploring the concept of a "take a break" reminder. The main idea (described in `structure.txt`) is to wait a set period, then automatically open a URL in the browser — prompting the user to step away from work. The scripts demonstrate `time.sleep()` and `webbrowser.open()` at different levels of completeness.

## Features

- Opens a YouTube video in the default web browser
- Uses timed delays before opening the URL
- `firstTry.py`: Repeats the open-and-wait cycle 3 times with 10-second intervals
- `openURL.py`: Opens a single URL immediately
- `wait2Hours.py`: Sleeps for 10 seconds (placeholder for a 2-hour wait)

## Project Structure

```
Take-A-Break/
├── firstTry.py
├── openURL.py
├── wait2Hours.py
├── structure.txt
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `time` and `webbrowser` from the standard library)

## Installation

```bash
cd "Take-A-Break"
```

No additional installation needed — standard library only.

## Usage

```bash
# Run the main loop (opens URL 3 times with 10s delays)
python firstTry.py

# Open the URL once immediately
python openURL.py

# Sleep for 10 seconds (demo/placeholder)
python wait2Hours.py
```

## How It Works

### `firstTry.py`
Loops 3 times: sleeps 10 seconds, then opens `https://www.youtube.com/watch?v=7wtfhZwyrcc` in the default browser.

### `openURL.py`
Opens `https://www.youtube.com/watch?v=7wtfhZwyrcc` in the default browser immediately (single call).

### `wait2Hours.py`
Calls `time.sleep(10)` — appears to be a work-in-progress placeholder for a 2-hour wait (the name suggests 2 hours, but the actual sleep is 10 seconds).

### `structure.txt`
Describes the intended design:
1. Get/set favorite URLs
2. Measure 2 hours of elapsed time
3. Open the browser at one of the set URLs
4. Loop this behavior

## Configuration

- The YouTube URL is hardcoded in `firstTry.py` and `openURL.py`: `https://www.youtube.com/watch?v=7wtfhZwyrcc`
- The sleep duration is hardcoded: 10 seconds in `firstTry.py` and `wait2Hours.py`
- The loop count is hardcoded: 3 iterations in `firstTry.py`

## Limitations

- The project is incomplete — `structure.txt` describes features (configurable URLs, 2-hour timer, looping) that are not implemented
- `wait2Hours.py` only sleeps 10 seconds despite its name
- No way to configure URLs or timing without editing source code
- No user interface or notification — just browser tabs opening
- The hardcoded YouTube URL may become unavailable

## Security Notes

No security concerns identified.

## License

Not specified.
