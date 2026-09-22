# Geocoding

> A Python script that converts a street address into geographic coordinates with the LocationIQ geocoding API.

## Overview

This script accepts an address, sends it to the LocationIQ geocoding API, and returns the corresponding latitude and longitude.

## Features

- Converts any address to latitude/longitude coordinates (geocoding)
- Uses the LocationIQ REST API (`us1.locationiq.com`)
- Interactive or command-line address input
- Displays both latitude and longitude from the API response

## Project Structure

```
Geocoding/
├── geocoding.py       # Main geocoding script
├── pyproject.toml     # Project metadata and dependencies
└── uv.lock            # Locked dependency versions
```

## Requirements

- Python 3.13+
- `requests`, managed by uv in `pyproject.toml`
- A LocationIQ API token (free tier available)

## Installation

```bash
cd "Geocoding"
uv sync
```

## Usage

1. Create a free account at [LocationIQ](https://locationiq.com/) and get your private token.
2. Run:

```bash
uv run python geocoding.py
```

3. Enter the API token only when prompted. The token is not echoed.

You can also provide both values directly:

```bash
uv run python geocoding.py "1600 Amphitheatre Parkway, Mountain View, CA" --token your-token
```

Example interaction:

```
Input the address: 1600 Amphitheatre Parkway, Mountain View, CA
The latitude of the given address is: 37.4224764
The longitude of the given address is: -122.0842499
Thanks for using this script
```

## How it works

1. Prompts for an address and hidden API token when they are not supplied as arguments.
2. Sends a bounded request with the API key, query address, and `format=json`.
3. Extracts `lat` and `lon` from the first result or reports a clear failure.

## Configuration

| Option | Description |
|---|---|
| `--token` | LocationIQ API token; omitted values are requested through a hidden prompt |

## Limitations

- Always takes the first result from the API response — may not be the most accurate for ambiguous addresses
- No rate limiting consideration (LocationIQ free tier has request limits)

## Security Notes

- The token is not stored in source code. Do not share or commit it to version control.
- Supplying a token with `--token` may expose it in shell history; use the interactive prompt when that matters.

## License

Not specified.
