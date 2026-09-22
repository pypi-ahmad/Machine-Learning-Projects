# CLI Proxy Tester

## Overview

A command-line tool for testing HTTP, HTTPS, and SOCKS proxy servers. It validates proxies by sending requests through them to an IP-checking service and comparing the returned IP with the proxy address. Results are stored in a CSV file.

**Type:** CLI Tool

## Features

- Test individual proxies (HTTP, HTTPS, SOCKS4, SOCKS5)
- Bulk test all proxies in a CSV file
- Import and test proxies from a plain text file
- Proxy validation via regex (IPv4 addresses and domain URLs)
- Automatic proxy deduplication in CSV storage
- Status tracking: "Proxy functional", "Proxy not functional", "Invalid response", "Proxy error"
- Customizable IP test service endpoint
- Includes a PHP script to self-host the IP test service
- Logging via Python's `logging` module

## Dependencies

Managed by uv in `pyproject.toml` and locked in `uv.lock`:

| Package  | Purpose |
|----------|---------|
| click    | Command-line interface |
| pandas   | CSV storage and updates |
| requests | Proxy HTTP requests |

Additional standard library imports: `re`, `logging`, `json.decoder`, `pathlib`

## How it works

### CLI Layer (`cli.py`)
- Built with the `click` library using a command group pattern.
- **`single`** command: Takes a proxy string (e.g., `http://1.1.1.1`), validates it with a regex, tests it, and saves results to CSV.
- **`csv_file`** command: Re-tests all proxies in an existing CSV file.
- **`add_from_txt_file`** command: Reads proxies line-by-line from a text file, tests each, and saves to CSV.
- Proxy format is validated via regex supporting IPv4 and domain-based addresses with protocol prefixes.

### Proxy Testing Layer (`proxytest.py`)
- `test_proxy()`: Sends a GET request through the proxy to the IP test service. Compares the returned JSON IP against the proxy address to determine functionality.
- `add_proxies_to_file()`: Manages a CSV file using `pandas`. Creates the file if it doesn't exist, updates existing proxy statuses, and appends new entries. Deduplicates before saving.
- `test_single_proxy()`: Splits the proxy string into type and address, tests it, and saves the result.
- `test_csv_file()`: Iterates through all proxies in a CSV file and re-tests each one.
- `add_from_text_file()`: Reads proxies from a text file and tests each individually.

### IP Test Service (`ipinfo/index.php`)
- A minimal PHP script that returns a JSON response containing the client's IP, X-Forwarded-For header, and User-Agent.

## Project Structure

```
cli_proxy_tester/
├── cli.py              # Click-based CLI interface
├── proxytest.py        # Core proxy testing and CSV management logic
├── pyproject.toml      # Python project and dependency metadata
├── uv.lock             # Reproducible dependency lock file
├── .gitignore          # Git ignore rules (excludes proxies.csv)
├── ipinfo/
│   └── index.php       # Self-hosted IP test service (PHP)
└── README.md
```

## Setup & Installation

```powershell
cd "Python Scripts\CLI Proxy Tester"
uv sync
```

## How to Run

### Test a single proxy

```powershell
uv run python cli.py single http://1.1.1.1
```

### Test with a custom IP test service

```powershell
uv run python cli.py single http://1.1.1.1 --iptest iptest.yourdomain.com
```

### Re-test all proxies in a CSV file

```powershell
uv run python cli.py csv-file proxies.csv
```

### Import and test proxies from a text file

```powershell
uv run python cli.py add-from-txt-file proxy_candidates.txt
```

Each text file line should be in the format `protocol://address` (e.g., `http://1.2.3.4:8080`).

## Configuration

- **`--iptest`**: Custom IP test service URL (default: `iptest.ingokleiber.de`). Available on all commands.
- **`--csv`**: Custom CSV file path for storing results (default: `proxies.csv`). Available on `single` and `add-from-txt-file` commands.
- To self-host the IP test service, deploy `ipinfo/index.php` on a web server accessible via both HTTP and HTTPS.

## Testing

No formal test suite present.

## Limitations

- The proxy regex validator does not support authentication (user:pass@host) in proxy URLs.
- No timeout configuration for proxy test requests — slow or unresponsive proxies may hang indefinitely.
- No concurrent/parallel proxy testing — proxies are tested sequentially.
- The IP test service (`ipinfo/index.php`) requires a PHP-capable web server.

## Security Notes

- Proxy addresses and test results are stored in a plaintext CSV file (`proxies.csv`).
- No authentication credentials are stored, but proxy addresses may be sensitive in some use cases.
- The PHP IP test service exposes client IP and User-Agent information — ensure it is not publicly accessible if used for private testing.
