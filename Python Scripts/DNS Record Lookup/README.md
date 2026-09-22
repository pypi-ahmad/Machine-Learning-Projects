# DNS Record Fetcher

> A CLI script that fetches DNS A and MX records for a given website using `dnspython`.

## Overview

This script queries DNS A records (IP addresses) and MX records (mail exchange servers) for a domain and prints them to the console.

## Features

- Fetches the DNS **A record** (IP address) for a domain
- Fetches all DNS **MX records** (mail servers) for a domain
- Accepts an optional domain argument or interactive input
- Shows DNS lookup failures without a traceback

## Project Structure

```
Dns_record/
├── dns_record.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Requirements

- Python 3.13+
- `dnspython`, managed by uv in `pyproject.toml`

## Installation

```bash
cd Dns_record
uv sync
```

## Usage

```bash
uv run python dns_record.py
```

Or pass the domain directly:

```bash
uv run python dns_record.py google.com
```

When prompted, enter a domain name (e.g., `google.com`):

```
Enter the name of the website: google.com
```

Sample output:

```
A records:
142.250.80.46
MX records:
10 smtp.google.com.
```

## How It Works

1. Uses `dns.resolver.resolve()` to query all **A records**.
2. Queries all **MX records**.
3. Prints each record grouped by type.

## Configuration

No configuration files. Provide the domain as an argument or interactively at runtime.

## Limitations

- Only A and MX records are queried.

## License

Not specified.
