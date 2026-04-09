# Fetch HTTP Status Code

> A command-line tool that fetches and displays the HTTP status code of a given URL, decorated with emoji indicators.

## Overview

This script prompts the user for a URL or API endpoint, sends an HTTP request using Python's `urllib`, and displays the resulting status code along with a thumbs-up or thumbs-down emoji to indicate success or failure.

## Features

- Fetches HTTP status codes for any URL or API endpoint
- Displays success responses with a thumbs-up emoji and the status code
- Displays HTTP error responses with a thumbs-down emoji, error code, and reason
- Handles URL/connection errors separately with descriptive messages
- Uses the `emoji` library for terminal-friendly emoji rendering

## Project Structure

```
Fetch HTTP status code/
├── fetch_http_status_code.py   # Main script
└── requirements.txt            # Python dependencies
```

## Requirements

- Python 3.x
- `emoji==0.6.0`

## Installation

```bash
cd "Fetch HTTP status code"
pip install -r requirements.txt
```

## Usage

```bash
python fetch_http_status_code.py
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

1. Prompts the user to enter a URL.
2. Calls `urllib.request.urlopen()` to send the request.
3. On success: prints the HTTP status code with a thumbs-up emoji and the response reason.
4. On `HTTPError`: prints the HTTP error code with a thumbs-down emoji and the error reason.
5. On `URLError`: parses the error reason string to extract the Windows error code and message, displays with a thumbs-down emoji.

## Configuration

No configuration needed.

## Limitations

- Uses `urlopen` which only sends GET requests — cannot test POST, PUT, DELETE, etc.
- The `URLError` parsing logic uses string splitting on `]` and `[`, which is fragile and Windows-specific
- The `emoji` library version `0.6.0` is very old and may not work on newer Python versions
- No timeout configured for the request — may hang on unresponsive servers
- The URL must include the protocol scheme (e.g., `https://`)

## Security Notes

No sensitive credentials in the code.

## License

Not specified.
