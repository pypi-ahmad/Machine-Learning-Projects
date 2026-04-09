# Internet Connection Check

> A Python script that checks internet connectivity by attempting to reach Google's homepage.

## Overview

This script tests whether the machine has an active internet connection by sending an HTTP GET request to `https://www.google.com/` with a 10-second timeout. It prints the connection status and returns a boolean result.

## Features

- Checks internet connectivity via an HTTP GET request to Google
- Configurable timeout (currently set to 10 seconds)
- Returns `True` if connected, `False` otherwise
- Prints descriptive status messages during the check

## Project Structure

```
Internet_connection_check/
├── internet_connection_check.py
└── output.png
```

## Requirements

- Python 3.x
- `requests`

## Installation

```bash
cd "Internet_connection_check"
pip install requests
```

## Usage

```bash
python internet_connection_check.py
```

**Expected output (connected):**
```
Attempting to connect to https://www.google.com/ to determine internet connection status.
Connection to https://www.google.com/ was successful.
```

**Expected output (disconnected):**
```
Attempting to connect to https://www.google.com/ to determine internet connection status.
Failed to connect to https://www.google.com/.
```

## How It Works

1. The `internet_connection_test()` function sends a GET request to `https://www.google.com/` using the `requests` library.
2. A 10-second timeout is applied to the request.
3. If the request succeeds, it prints a success message and returns `True`.
4. If any exception occurs, it prints a failure message and returns `False`.
5. The function is called automatically when the script is run.

## Configuration

- **Target URL:** Hardcoded as `https://www.google.com/` in the `internet_connection_test()` function.
- **Timeout:** Hardcoded to 10 seconds in the `requests.get()` call.

## Limitations

- The `except` block catches all exceptions with a bare `except:`, not just `requests.ConnectionError`. The `requests.ConnectionError` on the next line is a standalone expression that does nothing.
- Only tests connectivity to Google — if Google is down but other sites are accessible, it will incorrectly report no connection.
- The target URL and timeout are hardcoded and cannot be configured without editing the source.
- The function is called at module level, so importing this module will trigger the connectivity check.

## Security Notes

No security concerns identified.

## License

Not specified.
