# Shutdown or Restart Your Device

> A cross-platform command-line script to shut down or restart your computer using Python.

## Overview

This script prompts the user for a single-character input (`r` or `s`) and then executes the appropriate OS-level shutdown or restart command. It supports Windows, Linux, and macOS.

## Features

- Shutdown the computer on Windows, Linux, or macOS
- Restart the computer on Windows, Linux, or macOS
- Cross-platform detection using `platform.system()`
- Single-character interactive prompt

## Project Structure

```
Shutdown_or_restart_your_device/
└── PowerOptions.py
```

## Requirements

- Python 3.x
- No third-party dependencies (uses only `os`, `platform`)

## Installation

```bash
cd "Shutdown_or_restart_your_device"
# No additional packages required
```

## Usage

```bash
python PowerOptions.py
```

**Interactive prompt:**

```
Use 'r' for restart and 's' for shutdown: s
```

- Enter `r` to restart the machine
- Enter `s` to shut down the machine

> **Warning:** This script will immediately shut down or restart your computer. Save all work before running.

## How It Works

1. `platform.system()` detects the operating system
2. Based on user input:
   - **Shutdown:**
     - Windows: `os.system('shutdown -s')`
     - Linux/macOS: `os.system("shutdown -h now")`
   - **Restart:**
     - Windows: `os.system("shutdown -t 0 -r -f")` (immediate forced restart)
     - Linux/macOS: `os.system('reboot now')`
3. If the OS is not recognized, prints "Os not supported!"

## Configuration

No configuration needed.

## Limitations

- No confirmation prompt before executing — the command runs immediately
- The Windows shutdown command (`shutdown -s`) has a default 30-second delay; only the restart uses `-t 0` for immediate action
- On Linux/macOS, `reboot now` and `shutdown -h now` typically require root/sudo privileges
- No timer or scheduled shutdown option
- Input is case-insensitive via `.lower()`, but only single characters `r` and `s` are accepted

## Security Notes

- Requires elevated privileges on Linux/macOS to execute shutdown/reboot commands
- On Windows, executing `shutdown` or `reboot` commands via `os.system()` can be disruptive; use with caution

## License

Not specified.
