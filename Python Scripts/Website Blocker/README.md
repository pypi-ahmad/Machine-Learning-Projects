# Website Blocker

> Blocks and unblocks websites by modifying the system hosts file to redirect specified domains to localhost.

## Overview

This project consists of two scripts: `website_blocker.py` adds entries to the system hosts file to redirect specified websites to `127.0.0.1`, effectively blocking them. `website_unblocker.py` reverses the process by removing those entries from the hosts file.

## Features

- Cross-platform hosts file detection (Windows and Linux)
- Blocks websites by appending redirect entries (`127.0.0.1`) to the hosts file
- Unblocks websites by filtering out matching lines from the hosts file
- Duplicate entry prevention — checks if a site is already blocked before adding

## Project Structure

```
Website_blocker/
├── README.md
├── website_blocker.py
└── website_unblocker.py
```

## Requirements

- Python 3.x
- No external dependencies — uses only the `platform` standard library module
- **Administrator/root privileges** required to modify the hosts file

## Installation

```bash
cd Website_blocker
```

No `pip install` needed — all imports are from the Python standard library.

## Usage

### Block websites

```bash
# Windows (run as Administrator)
python website_blocker.py

# Linux
sudo python website_blocker.py
```

### Unblock websites

```bash
# Windows (run as Administrator)
python website_unblocker.py

# Linux
sudo python website_unblocker.py
```

## How It Works

### website_blocker.py
1. Detects the OS using `platform.system()` and sets the hosts file path:
   - Windows: `C:\Windows\System32\drivers\etc\hosts`
   - Linux: `/etc/hosts`
2. Opens the hosts file in read-append mode (`'r+'`).
3. For each URL in the `websites` list, checks if it already exists in the file content.
4. If not present, appends a line: `127.0.0.1 <website_url>`.

### website_unblocker.py
1. Same OS detection for the hosts file path.
2. Opens the hosts file in read-write mode (`'r+'`).
3. Reads all lines, seeks back to the beginning.
4. Rewrites only the lines that do **not** contain any of the blocked websites.
5. Truncates the file to remove any leftover content.

## Configuration

The list of websites to block/unblock is hardcoded in the `websites` array in both scripts. To change which sites are blocked, edit this list:

```python
websites = ["https://example.com/", "https://another-site.com/"]
```

**Important**: The same list must be maintained in both `website_blocker.py` and `website_unblocker.py` for unblocking to work correctly.

## Limitations

- The website list is hardcoded and identical across both scripts — no shared configuration.
- No macOS support (only Windows and Linux paths are handled); other OS values from `platform.system()` are not handled, leaving `pathToHosts` undefined.
- Blocks full URLs (including `https://www.`) rather than just domain names, which may not match how the hosts file typically works (hosts file matches domain names, not full URLs).
- No command-line interface for specifying sites to block/unblock at runtime.
- No backup of the hosts file before modification.
- No error handling for permission denied or file not found.

## Security Notes

- Requires elevated privileges (Administrator on Windows, root on Linux) to modify the system hosts file.
- The hardcoded website list contains adult content URLs.

## License

Not specified.
