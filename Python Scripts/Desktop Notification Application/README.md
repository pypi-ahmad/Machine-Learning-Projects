# Desktop Notification Application

## Overview

A utility script that sends periodic desktop notifications reminding the user to take a break. It uses the `plyer` library to display OS-native notifications every hour in an infinite loop.

**Type:** Utility

## Features

- Sends a desktop notification with a customizable title and message
- Notification displays for 10 seconds (timeout of 10)
- Repeats every 3600 seconds (1 hour) in an infinite loop
- Cross-platform notification support via `plyer`

## Dependencies

- `plyer` — cross-platform desktop notifications
- `time` (Python standard library)

## How It Works

1. The script enters an infinite `while True` loop.
2. On each iteration, `notification.notify()` from `plyer` sends a desktop notification with the title "ALERT!!!" and message "Take a break! It has been an hour!".
3. The notification has a timeout of 10 seconds.
4. After sending the notification, the script sleeps for 3600 seconds (1 hour) before sending the next one.

## Project Structure

```
Desktop Notification Application/
└── Desktop Notification Application.py   # Main script
```

## Setup & Installation

1. Ensure Python 3.x is installed.
2. Install dependencies:
   ```bash
   pip install plyer
   ```

## How to Run

```bash
python "Desktop Notification Application.py"
```

The script will run indefinitely, sending a notification every hour. Press `Ctrl+C` to stop.

## Configuration

No external configuration file. To change the notification interval, title, or message, edit the values directly in the script:

- `title` — Notification title (default: `"ALERT!!!"`)
- `message` — Notification message (default: `"Take a break! It has been an hour!"`)
- `timeout` — How long the notification displays in seconds (default: `10`)
- `time.sleep(3600)` — Interval between notifications in seconds (default: `3600`)

## Testing

No formal test suite present.

## Limitations

- The script runs in an infinite loop and must be manually terminated (`Ctrl+C`).
- The notification title and message are hardcoded; there is no CLI argument or config file support.
- No logging or error handling if the notification system is unavailable.
- The sleep interval is blocking; the script cannot be interacted with between notifications.
