# Capture Screenshot

## Overview

A command-line utility that captures screenshots at regular intervals and saves them as JPEG files with timestamped filenames. Useful for time-lapse monitoring, activity logging, or automated screen capture.

**Type:** CLI Utility

## Features

- Configurable capture frequency (per hour, per minute, or per second)
- Customizable output directory via command-line argument
- Automatic directory creation if the output path does not exist
- Timestamped filenames in `HH_MM_SS.jpg` format
- Graceful shutdown via `Ctrl+C` (KeyboardInterrupt)
- Continuous capture loop until manually stopped

## Dependencies

From `requirements.txt`:

| Package    | Version |
|------------|---------|
| PyAutoGUI  | 0.9.50  |

Additional standard library imports: `os`, `argparse`, `time`

## How It Works

1. Command-line arguments are parsed using `argparse`:
   - `-p` / `--path`: Directory to store screenshots (default: `./images`)
   - `-t` / `--type`: Time unit — `h` (hours), `m` (minutes), or `s` (seconds) (default: `h`)
   - `-f` / `--frequency`: Number of screenshots per time unit (default: `1`)
2. The interval in seconds is calculated from the type and frequency. A minimum interval of 1 second is enforced.
3. If the output directory does not exist, it is created.
4. In an infinite loop, a screenshot is taken via `pyautogui.screenshot()`, saved with the current time as filename, and the script sleeps for the computed interval.
5. The loop is interrupted cleanly by pressing `Ctrl+C`.

## Project Structure

```
capture_screenshot/
├── screenshot.py       # Main script
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup & Installation

```bash
cd capture_screenshot
pip install -r requirements.txt
```

## How to Run

```bash
# Take 1 screenshot per hour (default), saved to ./images/
python screenshot.py

# Take 5 screenshots per minute, saved to ./images/
python screenshot.py -t m -f 5

# Take 1 screenshot per second, saved to custom directory
python screenshot.py -t s -f 1 -p /path/to/output

# Take 10 screenshots per minute, saved to custom directory
python screenshot.py -t m -f 10 -p ./my_screenshots
```

Press `Ctrl+C` to stop capturing.

## Configuration

No environment variables or config files required. All options are passed via command-line arguments.

## Testing

No formal test suite present.

## Limitations

- The `-t s` (seconds) mode ignores the frequency value — the interval calculation for seconds is not implemented; it falls through to the minimum of 1 second.
- Filenames use `HH_MM_SS` format, so multiple screenshots taken within the same second will overwrite each other.
- Only saves in JPEG format (`.jpg`).
- No maximum capture count option — runs indefinitely until interrupted.
- On some Linux systems, `pyautogui` requires additional dependencies (e.g., `scrot` or `gnome-screenshot`).
