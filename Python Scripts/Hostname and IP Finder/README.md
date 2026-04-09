# Find Hostname and IP Address

> A Python script that resolves a website's hostname to its IP address using the `socket` module.

## Overview

This script prompts the user for a website URL (hostname), performs a DNS lookup using Python's built-in `socket` module, and displays the resolved IP address.

## Features

- Resolves any hostname to its IP address via DNS lookup
- Handles invalid hostnames with a descriptive error message
- Uses Python's built-in `socket.gethostbyname()` — no external dependencies
- Interactive command-line prompt

## Project Structure

```
Find_out_hostname_and_ip_address/
├── Hostname_IPaddress.py   # Main script
└── Screenshot.png          # Sample output screenshot
```

## Requirements

- Python 3.x
- No external dependencies (uses only the built-in `socket` module)

## Installation

```bash
cd "Find_out_hostname_and_ip_address"
```

No package installation needed.

## Usage

```bash
python Hostname_IPaddress.py
```

Example interaction:

```
Please enter website address(URL): www.google.com
Hostname: www.google.com
IP: 142.250.80.4
```

Error example:

```
Please enter website address(URL): invalid.hostname.xyz
Invalid Hostname, error raised is [Errno 11001] getaddrinfo failed
```

## How It Works

1. Defines a function `get_hostname_IP()`.
2. Prompts the user for a website address.
3. Calls `socket.gethostbyname(hostname)` to resolve the hostname to an IPv4 address.
4. Prints the hostname and resolved IP.
5. Catches `socket.gaierror` if the hostname cannot be resolved and prints the error.

## Configuration

No configuration needed.

## Limitations

- Only resolves to IPv4 addresses (`gethostbyname` returns a single IPv4 address)
- Does not support looking up multiple IPs for a hostname (use `getaddrinfo` for that)
- The input expects a bare hostname (e.g., `www.google.com`), not a full URL with protocol
- No validation of input format

## Security Notes

No security concerns.

## License

Not specified.
