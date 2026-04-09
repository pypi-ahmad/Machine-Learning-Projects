# Fetch Open Ports

> A Python port scanner that checks for open TCP ports on a given host within a specified range.

## Overview

This script prompts the user for a hostname or IP address, resolves it to an IP, and scans TCP ports 50–500 to identify which ones are open. It reports results in real-time and displays the total scan time.

## Features

- Resolves hostnames to IP addresses using `gethostbyname()`
- Scans TCP ports in the range 50–500
- Reports open ports in real-time as they are discovered
- Displays total scan duration

## Project Structure

```
Fetch_open_ports/
├── fetch_open_port.py   # Main port scanning script
└── Screenshot.png       # Sample output screenshot
```

## Requirements

- Python 3.x
- No external dependencies (uses only the built-in `socket` and `time` modules)

## Installation

```bash
cd "Fetch_open_ports"
```

No package installation needed.

## Usage

```bash
python fetch_open_port.py
```

Example interaction:

```
Enter the host to be scanned: scanme.nmap.org
Starting scan on host: 45.33.32.156
Port 80: OPEN
Port 443: OPEN
Time taken: 42.31
```

## How It Works

1. Prompts the user for a target hostname.
2. Resolves the hostname to an IP address using `socket.gethostbyname()`.
3. Iterates through ports 50–500.
4. For each port, creates a TCP socket (`AF_INET`, `SOCK_STREAM`) and attempts a connection using `connect_ex()`.
5. If `connect_ex()` returns 0, the port is open and reported.
6. Closes each socket after the check.
7. Prints the total elapsed time after the scan completes.

## Configuration

No configuration files. The port range (50–500) is hardcoded in the script.

## Limitations

- Port range is hardcoded to 50–500 — cannot scan other ranges without modifying the code
- No timeout set on socket connections — each closed port may take several seconds to time out
- Sequential scanning (one port at a time) makes it slow for large ranges
- No UDP port scanning support
- No banner grabbing or service identification
- The `print('Time taken:...')` statement at the end has incorrect indentation and runs outside the `if __name__` block
- No error handling for unresolvable hostnames

## Security Notes

Port scanning may be considered intrusive or unauthorized on networks you don't own. Only scan hosts you have permission to test.

## License

Not specified.
