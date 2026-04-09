# Speed Test

> A Python script that measures internet speed by invoking the `speedtest-cli` command-line tool.

## Overview

This script runs the external `speedtest-cli` command via `subprocess.check_output()` and prints the results (download speed, upload speed, and ping) to the console.

## Features

- Measures download speed, upload speed, and ping
- Uses `subprocess.check_output()` to capture and display `speedtest-cli` output
- Single-file script with no interactive input

## Project Structure

```
Speed Test/
└── speedtest.py
```

## Requirements

- Python 3.x
- `speedtest-cli` (must be installed and available on the system PATH)

## Installation

```bash
cd "Speed Test"
pip install speedtest-cli
```

## Usage

```bash
python speedtest.py
```

**Sample Output:**

```
The Result of Speed Test
Retrieving speedtest.net configuration...
Testing from ISP (xxx.xxx.xxx.xxx)...
Retrieving speedtest.net server list...
Selecting best server based on ping...
Hosted by Server Name [Location]: xx.xxx ms
Testing download speed................
Download: xx.xx Mbit/s
Testing upload speed................
Upload: xx.xx Mbit/s
```

## How It Works

1. `subprocess.check_output("speedtest-cli", shell=True, universal_newlines=True)` executes the `speedtest-cli` command and captures its stdout as a string
2. The result is printed to the console with a header line

## Configuration

No configuration options. The script runs `speedtest-cli` with its default settings.

## Limitations

- Requires `speedtest-cli` to be installed as a system command — the script does not use the `speedtest` Python library directly
- Uses `shell=True` in `subprocess.check_output()`, which is generally discouraged for security reasons
- No error handling — if `speedtest-cli` is not installed, the script raises a `CalledProcessError`
- No option to select a specific server or output format
- The `speedtest-cli` package is deprecated in favor of `speedtest` by Ookla

## Security Notes

- `shell=True` in `subprocess.check_output()` could be a security risk if the command string were user-controlled, but in this case the command is hardcoded

## License

Not specified.
