# Pomodoro GUI

> A Tkinter-based Pomodoro timer with 25-minute work sessions, 5-minute breaks, website blocking, session reporting, and sound alerts.

## Overview

This application implements the Pomodoro Technique — a time management method that alternates between 25-minute focused work sessions and 5-minute breaks. It includes a website blocker (Windows only) that modifies the hosts file to block distracting sites during work sessions, a session reporting feature, and audio alerts when timers complete.

## Features

- **25-minute work timer**: Full-screen countdown display with alarm on completion
- **5-minute break timer**: Separate countdown with alarm notification
- **Website blocker**: Block distracting URLs by modifying the Windows hosts file
  - Add multiple URLs to block list
  - Unblock all sites when done
- **Session report**: Tracks and displays:
  - Number of Pomodoros completed
  - Number of breaks completed
  - Total hours of work
  - Total hours of break
- **Sound alerts**: Plays `beep.wav` when timers finish (via pygame)
- **Custom background image**: Uses `bg.png` for the main window

## Project Structure

```
Pomodoro_GUI/
├── beep.wav            # Alarm sound file
├── bg.png              # Background image
├── Pomodoro_gui.py     # Main application
└── requirements.txt    # Dependencies
```

## Requirements

- Python 3.x
- `pygame==2.0.1`
- `tkinter` (standard library)
- `datetime` (standard library)
- `time` (standard library)

## Installation

```bash
cd "Pomodoro_GUI"
pip install -r requirements.txt
```

Or manually:

```bash
pip install pygame==2.0.1
```

## Usage

```bash
python Pomodoro_gui.py
```

1. **WEBSITE BLOCKER**: Click to open the blocker popup. Enter URLs and click "Block". Click "Unblock all" when finished.
2. **START WORK TIMER**: Launches a 25-minute countdown. An alert and sound play when time is up.
3. **START BREAK TIMER**: Launches a 5-minute countdown. An alert and sound play when time is up.
4. **SHOW REPORT**: Displays session statistics (Pomodoros completed, break count, total work/break time).

## How It Works

1. **`main()`**: Creates the root Tkinter window with a background image and four buttons.
2. **`pomodoro_timer()`**: Opens a popup with a 25-minute countdown (`25*60` seconds). Updates the display every second via `time.sleep(1)`. On completion, shows a messagebox and plays `beep.wav`.
3. **`break_timer()`**: Same as above but for 5 minutes (`5*60` seconds).
4. **`block_websites()`**: Appends URLs to the Windows hosts file (`C:\Windows\System32\drivers\etc\hosts`) with redirect to `127.0.0.1`.
5. **`remove_websites()`**: Reads the hosts file, removes previously added entries, and truncates the file.
6. **`show_report()`**: Calculates total work/break time from `pomo_count` using `timedelta`. Note: uses `pomo_count` for both work and break calculations (does not use `break_count`).

## Configuration

- **Hosts file path**: `C:\Windows\System32\drivers\etc\hosts` (hardcoded, Windows only)
- **Work duration**: 25 minutes (hardcoded in `pomodoro_timer()`)
- **Break duration**: 5 minutes (hardcoded in `break_timer()`)
- **Sound file path**: `./Pomodoro_GUI/beep.wav` (relative path — must be run from parent directory)
- **Background image**: `./Pomodoro_GUI/bg.png` (same path requirement)
- **Window size**: 470×608 pixels

## Limitations

- **Windows only**: Website blocker modifies the Windows hosts file and requires admin privileges.
- **Blocking UI**: `time.sleep(1)` in the timer loop freezes the main window during countdown.
- File paths for `beep.wav` and `bg.png` use `./Pomodoro_GUI/` prefix — the script must be run from the parent directory, not from within the `Pomodoro_GUI` folder.
- Bare `except: pass` clauses silently swallow all errors in timers and website unblocking.
- Timer durations are hardcoded — no custom duration support.
- Session data is not persisted between app restarts.
- Report calculation uses `pomo_count*25` for work and `pomo_count*5` for break (uses `pomo_count` for both, not `break_count` for breaks).
- `requirements.txt` contains `tkinter==8.6` which is not pip-installable; `pip install -r requirements.txt` will fail. Use `pip install pygame==2.0.1` instead.

## Security Notes

- **Hosts file modification**: The website blocker modifies a system file (`hosts`), requiring administrator/elevated privileges. Changes persist after the app closes if not unblocked.
- **PermissionError handling**: The app warns the user to run as admin if it can't write to the hosts file.

## License

Not specified.
