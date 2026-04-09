# URL Shortener

> A command-line URL shortener using the TinyURL API.

## Overview

A script that takes one or more URLs as command-line arguments and returns shortened versions using the TinyURL API (`http://tinyurl.com/api-create.php`).

## Features

- Shortens URLs via the TinyURL public API
- Accepts multiple URLs as command-line arguments
- Uses `contextlib.closing` for proper resource cleanup
- Prints shortened URLs to stdout

## Project Structure

```
URL_SHORTENER/
├── app.py
└── README.md
```

## Requirements

- Python 3.x (with fixes — see Limitations)
- No external dependencies (uses only standard library modules)

## Installation

```bash
cd "URL_SHORTENER"
```

No package installation required.

## Usage

**Note:** The script has import errors that must be fixed before it can run (see Limitations).

Intended usage:

```bash
python app.py https://www.example.com https://www.google.com
```

Each URL argument is shortened and the result is printed.

## How It Works

1. Takes URLs from `sys.argv[1:]` (command-line arguments after the script name).
2. For each URL, calls `short_url()` which:
   - Constructs a request to `http://tinyurl.com/api-create.php` with the URL as a query parameter.
   - Opens the request URL and reads the response (the shortened URL).
   - Returns the decoded UTF-8 response string.
3. Prints each shortened URL.

## Configuration

No configuration needed. The TinyURL API endpoint is hardcoded.

## Limitations

- **Import errors:** The script has conflicting imports that will cause runtime failures:
  - `from urllib.parse import urlencode` (Python 3) followed by `from urllib import urlencode` (Python 2) — the second import will fail in Python 3.
  - `from urllib.request import urlopen` (Python 3) followed by `from urllib2 import urlopen` (Python 2) — `urllib2` does not exist in Python 3.
- The `response.read().decode('utf-8 ')` has a trailing space in the encoding name which may cause a `LookupError`.
- No error handling for network failures, invalid URLs, or API downtime.
- Depends on the TinyURL public API remaining available and free.
- `from __future__ import with_statement` is unnecessary in Python 3 (it was needed for Python 2.5).

## License

Not specified.
