# Website Connectivity Checker

A command-line tool that checks whether HTTP(S) URLs are reachable. It supports individual URLs, URL files, table or JSON output, and optional CSV export.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

## Install

```powershell
cd "Python Scripts\Website Connectivity Checker"
uv sync
```

## Usage

Check one URL:

```powershell
uv run check_site https://example.com
```

Check several URLs:

```powershell
uv run check_site https://example.com https://pypi.org --timeout 5 --retries 1
```

Read URLs from a text or CSV file:

```powershell
uv run check_site --file .\urls.txt
```

Produce JSON or save CSV output:

```powershell
uv run check_site --json https://example.com
uv run check_site --csv .\results.csv https://example.com
```

## Exit codes

- `0`: Every checked URL returned a 2xx or 3xx response.
- `1`: At least one URL was unreachable or returned a non-success status.
- `2`: Input was missing or invalid.

## Notes

- Requests use a 10-second timeout and two retries by default.
- Run checks only against URLs you are authorized to probe.
- The project uses uv for dependency resolution and the native `uv_build` backend for packaging.
