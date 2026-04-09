# Site Blocker

> A Python script that blocks websites by modifying the Windows hosts file during specified hours.

## Overview

This script blocks access to a predefined list of websites by appending entries to the Windows `hosts` file, redirecting them to `127.0.0.1`. It is intended to run as a background process and enforces blocking during a configured time window (9 AM – 6 PM by default).

## Features

- Blocks websites by redirecting them to `127.0.0.1` via the hosts file
- Time-based blocking between 9:00 AM and 6:00 PM
- Removes blocked entries from the hosts file outside the blocking window
- Continuous monitoring loop
- Supports multiple websites in a configurable list

## Project Structure

```
Site-blocker/
└── web-blocker.py
```

## Requirements

- Python 3.x
- No third-party dependencies (uses only `time`, `datetime`)
- **Windows OS** (hardcoded hosts file path: `C:\Windows\System32\drivers\etc\hosts`)
- **Administrator privileges** required to modify the hosts file

## Installation

```bash
cd "Site-blocker"
# No additional packages required
```

## Usage

Run with administrator privileges:

```bash
python web-blocker.py
```

The script runs in an infinite loop. Press `Ctrl+C` to stop.

## How It Works

1. Defines a hosts file path (`C:\Windows\System32\drivers\etc\hosts`) and a redirect IP (`127.0.0.1`)
2. A list of websites to block is hardcoded: `www.amazon.in`, `www.youtube.com`, `youtube.com`, `www.facebook.com`, `facebook.com`
3. In a `while True` loop:
   - Prints "Access denied to Website" if the current time is between 9 AM and 6 PM (this check does not gate any subsequent logic)
   - Opens the hosts file and appends `127.0.0.1 <site>` for each site not already present
   - Via a `for/else` clause, re-reads the hosts file and attempts to remove blocked-site lines (runs unconditionally on every iteration)
4. `time.sleep(5)` is placed outside the `while True:` loop and never executes

## Configuration

Edit these values directly in `web-blocker.py`:

- **`hostsPath`**: Path to the hosts file (default: `C:\Windows\System32\drivers\etc\hosts`)
- **`redirect`**: Redirect IP address (default: `127.0.0.1`)
- **`websites`**: List of domains to block
- **Blocking hours**: Currently hardcoded as 9:00 to 18:00 — change the `dt(...)` comparisons in the `if` statement

## Limitations

- **Windows only** — the hosts file path is hardcoded for Windows
- The time-based `if` condition prints "Access denied to Website" but the blocking logic runs regardless of the time check due to indentation — the `with open(...)` block is outside the `if` block
- The `else` clause on the `for` loop causes the unblocking logic to execute on every iteration, not just outside blocking hours
- No graceful shutdown mechanism — must be killed with `Ctrl+C`
- The script does not validate that it has write access to the hosts file before attempting modifications
- `file.truncate()` is called inside the `for` loop instead of after it, potentially causing incorrect file contents

## Security Notes

- Requires **administrator/root privileges** to modify the system hosts file
- Modifying the hosts file affects all applications on the system, not just the browser
- The script can be bypassed by using DNS-over-HTTPS or alternative DNS resolvers

## License

Not specified.
