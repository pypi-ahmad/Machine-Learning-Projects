# Fetch HTTP Status Code

> A command-line tool that fetches and displays the HTTP status code of a given URL, decorated with emoji indicators.

## Overview

This script prompts the user for a URL or API endpoint, sends an HTTP request using Python's `urllib`, and displays the resulting status code along with a thumbs-up or thumbs-down emoji to indicate success or failure.

## Features

- Fetches HTTP status codes for any URL or API endpoint
- Displays success responses with a thumbs-up emoji and the status code
- Displays HTTP error responses with a thumbs-down emoji, error code, and reason
- Handles URL/connection errors separately with descriptive messages
- Uses standard-library Unicode indicators

## Project Structure

```
Fetch HTTP status code/
├── fetch_http_status_code.py   # Main script
├── pyproject.toml              # Project metadata and dependencies
└── uv.lock                     # Locked dependency versions
```

## Requirements

- Python 3.13+
- No third-party dependencies

## Installation

```bash
cd "Fetch HTTP status code"
uv sync
```

## Usage

```bash
uv run python fetch_http_status_code.py
```

Or pass the URL directly:

```bash
uv run python fetch_http_status_code.py https://www.google.com
```

Example interaction:

```
Enter the URL to be invoked: https://www.google.com
Status code : 200 👍
Message : Request succeeded. Request returned message - OK
```

```
Enter the URL to be invoked: https://httpstat.us/404
Status : 404 👎
Message : Request failed. Request returned reason - Not Found
```

## How It Works

1. Gets a URL from the command line or prompt.
2. Validates the HTTP(S) scheme and calls `urllib.request.urlopen()` with a timeout.
3. Prints the status and reason, handling HTTP and connection errors separately.

## Configuration

No configuration needed.

## Limitations

- Uses `urlopen`, so it checks GET requests only.
- The URL must include the protocol scheme (for example, `https://`).

## Security Notes

No sensitive credentials in the code.

## License

Not specified.
