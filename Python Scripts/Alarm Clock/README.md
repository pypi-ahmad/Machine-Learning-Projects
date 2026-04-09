# Set Alarm

> A command-line alarm clock that plays music from a local collection at a user-specified time (Windows only).

## Overview

This script lets you set an alarm by entering a time in `HH:MM` format. When the current system time reaches the alarm time, it opens a music file from the `musics/` folder using the Windows `start` command. You can choose from multiple alarm sounds if more than one is available.

## Features

- Set alarm time in `HH:MM` format with input validation via regex
- Automatically lists available music files from the `musics/` directory
- Lets user select an alarm sound by index (or auto-selects if only one exists)
- Renames music files with spaces to underscores for compatibility
- Displays a formatted header in the terminal
- Plays the alarm sound by launching it with the default Windows media handler (`cmd /C start`)

## Project Structure

```
Set_Alarm/
├── alarm.py
└── musics/
    ├── Carnival.ogg
    ├── Crusade.ogg
    ├── dreamy_nights.ogg
    ├── lakhau_hajarau.mp3
    ├── morning_calm.ogg
    ├── mozart_wakes.ogg
    ├── new_horizon.ogg
    ├── Renaissance.ogg
    ├── the_four_seasons.ogg
    └── Variations.ogg
```

## Requirements

- Python 3.x
- No third-party dependencies (uses only `datetime`, `os`, `re`, `subprocess`)
- **Windows OS** (uses `cmd /C start` to play audio)

## Installation

```bash
cd "Set_Alarm"
# No additional packages required
```

## Usage

```bash
python alarm.py
```

**Interactive prompts:**

1. Enter the alarm time in `HH:MM` format (e.g., `07:30`)
2. If multiple music files exist, select one by entering its index number
3. Keep the program running — the alarm triggers when the system clock reaches the set time

### Sample Output

```
       ###########################
       ###### Alarm Program ######
       ###########################

Set the alarm time (e.g. 01:10): 07:30

Select any alarm music:

1. Carnival
2. Crusade
3. Dreamy Nights
...

Enter the index of the listed musics (e.g. 1): 3
>> Alarm music has been set --> Dreamy Nights

>>> Alarm has been set successfully for 07:30! Please dont close the program! <<<
```

## How It Works

1. `display_header()` prints a centered ASCII header in the terminal
2. `set_alarm()` prompts the user for a time and validates it with `re.match(r"^[0-9]{2}:[0-9]{2}$", ...)`
3. Music files in the `musics/` subdirectory are listed; filenames with spaces are renamed to use underscores via `rename_files_with_whitespaces()`
4. The user selects a music file (or it auto-selects if only one exists)
5. A `while` loop continuously compares `datetime.datetime.now().time()` (as a string) against the target time
6. When the current time exceeds the alarm time, `subprocess.run(('cmd', '/C', 'start', ...))` launches the music file

## Configuration

- **Music files**: Place `.ogg` or `.mp3` files in the `musics/` subdirectory
- **Time format**: Only `HH:MM` (24-hour) is accepted

## Limitations

- **Windows only** — uses `cmd /C start` to play audio; will not work on Linux/macOS
- Time comparison is string-based (`current_time >= playback_time`), which works but is not timezone-aware
- The alarm cannot be set for the next day — if the set time has already passed today, the alarm triggers immediately
- Bare `except` clause catches all exceptions silently when the user enters an invalid music index
- Busy-wait loop with no `time.sleep()` — continuously polls `datetime.now()`, consuming CPU

## Security Notes

No sensitive data or security concerns identified.

## License

Not specified.
