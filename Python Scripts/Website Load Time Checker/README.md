# Time to Load Website

> A command-line tool that measures the time taken to load (read) a given URL.

## Overview

This script accepts a URL from the user and measures how long it takes to read the content of that URL using Python's `urllib`. It timestamps before and after the read operation to calculate loading time.

## Features

- Measures website load (read) time in seconds
- Automatically prepends `https://` if no protocol is provided
- Formatted output to 2 significant figures
- Well-documented function with docstring

## Project Structure

```
Time_to_load_website/
├── time_to_load_website.py
├── sample.PNG
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `urllib.request` and `time` from the standard library)

## Installation

```bash
cd "Time_to_load_website"
```

No package installation required.

## Usage

```bash
python time_to_load_website.py
```

When prompted, enter a URL:

```
Enter the url whose loading time you want to check: https://www.google.com
The time taken to load https://www.google.com is 0.25 seconds.
```

The `get_load_time()` function can also be imported and used programmatically:

```python
from time_to_load_website import get_load_time
load_time = get_load_time("https://www.example.com")
```

## How It Works

1. Takes the user-supplied URL.
2. Checks if the URL contains `http` or `https`; if not, prepends `https://`.
3. Opens the URL with `urlopen()`.
4. Records a timestamp, reads the full response, records another timestamp.
5. Returns the difference as the load time in seconds.

## Configuration

No configuration needed.

## Limitations

- The protocol detection logic `("https" or "http") in url` is flawed — it evaluates as `"https" in url` due to Python's truthy string evaluation, so the `http` check is never reached. URLs starting with `http://` (without `s`) won't get the fallback prepend but will still work since `http` contains the substring `https` is not matched — however, `http://` URLs will work fine since they already have a protocol.
- Measures only the read time after the connection is already open, not the full connection + read time.
- No error handling for invalid URLs, network errors, or timeouts.
- Output uses `:.2` format specifier (2 significant figures) rather than `:.2f` (2 decimal places).

## License

Not specified.
