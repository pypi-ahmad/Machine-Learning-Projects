# Check Website Connectivity

A modern CLI tool to check whether websites are reachable.

Replaces the legacy `check_connectivity.py` script with a fully-featured,
tested, and type-safe implementation.

## What changed from the legacy script

| Aspect               | Legacy (`check_connectivity.py`)          | Modern (`check_site`)                          |
| -------------------- | ----------------------------------------- | ---------------------------------------------- |
| HTTP library         | `requests`                                | `httpx` (modern, async-ready, timeout support) |
| Input                | Hardcoded `websites.txt`                  | CLI args, `--file` (txt/csv)                   |
| Output               | Hardcoded `website_status.csv`            | Table (default), `--json`, `--csv <path>`      |
| Timeout              | None (hangs indefinitely)                 | 10 s default, configurable `--timeout`         |
| Retries              | None                                      | 2 retries default, configurable `--retries`    |
| User-Agent           | Default `python-requests`                 | Custom descriptive UA string                   |
| Error handling       | Crashes on connection errors              | Catches timeouts, DNS, connection errors       |
| URL validation       | None                                      | Scheme normalisation + validation              |
| Logging              | None                                      | `--verbose` enables debug logging              |
| Exit codes           | Always 0                                  | 0=all ok, 1=any unreachable, 2=input error    |
| Status granularity   | "working" / "not working"                 | reachable / unreachable / error + HTTP code    |
| Tests                | None                                      | 40+ pytest tests with respx mocking            |
| Packaging            | `requirements.txt`                        | `pyproject.toml` (PEP 621) + src layout        |
| Global state         | Mutable module-level dict                 | No global mutable state                        |

## Installation

```bash
cd Check_website_connectivity
pip install -e .
```

## Usage

### Check a single URL

```bash
check_site https://example.com
```

### Check multiple URLs

```bash
check_site https://example.com https://github.com https://pypi.org
```

### Check URLs from a file

```bash
# Plain text, one URL per line:
check_site --file urls.txt

# CSV (reads URLs from first URL-like column, skips headers):
check_site --file website_status.csv
```

### JSON output

```bash
check_site --json https://example.com
```

### CSV output

```bash
check_site --csv results.csv https://example.com https://github.com
```

### Combine options

```bash
check_site --file urls.txt --csv out.csv --timeout 5 --retries 3 --verbose
```

## Global options

| Option                | Description                    | Default |
| --------------------- | ------------------------------ | ------- |
| `--file`, `-f`        | Path to a .txt or .csv file    | -       |
| `--json`              | Print results as JSON          | off     |
| `--csv PATH`          | Write results to a CSV file    | -       |
| `--timeout`, `-t`     | Request timeout in seconds     | 10.0    |
| `--retries`, `-r`     | Retry count on transient errors| 2       |
| `--verbose`, `-v`     | Enable debug logging           | off     |

## Exit codes

| Code | Meaning                                  |
| ---- | ---------------------------------------- |
| 0    | All URLs reachable (2xx/3xx)             |
| 1    | At least one URL unreachable or errored  |
| 2    | Invalid input (bad URL, missing file)    |

## Development

```bash
pip install -e ".[dev]"     # or: pip install -e . && pip install ruff pytest respx
ruff check src tests
ruff format --check src tests
pytest -q
```

## Project structure

```
Check_website_connectivity/
  pyproject.toml
  README.md
  src/check_website_connectivity/
    __init__.py        # version
    models.py          # CheckResult dataclass + Status enum
    core.py            # URL parsing, validation, HTTP checks, output formatting
    cli.py             # Typer CLI entry point
  tests/
    test_core.py       # Unit tests: parsing, validation, HTTP mocking, formatting
    test_cli.py        # CLI integration tests via CliRunner + respx
```
