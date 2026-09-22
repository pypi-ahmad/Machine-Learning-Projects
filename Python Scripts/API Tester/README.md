# API Tester

`main.py` is a standard-library command-line client for `http` and `https`
endpoints. It supports request methods, custom headers, JSON or form payloads,
response formatting, and a small local request-history summary.

## Install and run

```powershell
cd "Python Scripts/API Tester"
uv sync
uv run python main.py GET https://api.example.com/status
uv run python main.py POST https://api.example.com/items -d '{"name":"Alice"}'
uv run python main.py GET https://api.example.com/items -H "Accept: application/json" --no-history
```

Run without a method and URL to open the interactive prompt:

```powershell
uv run python main.py
```

Use `-v` to display response headers and `-t SECONDS` to set a positive request
timeout. The tool accepts only absolute `http://` or `https://` URLs.

## History and sensitive data

By default, `api_history.json` stores up to 200 local request summaries: method,
URL, timestamp, status, elapsed time, and any error. It does not store request
headers, request bodies, response headers, or response bodies. Use
`--no-history` to prevent even the summary from being written.

Avoid placing credentials directly in shell history. Prefer environment-backed
tools or interactive prompts when an API requires a secret.

The project uses only the Python standard library.
