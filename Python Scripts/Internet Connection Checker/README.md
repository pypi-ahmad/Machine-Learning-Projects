# Internet Connection Checker

Check whether an HTTPS URL responds within a chosen timeout.

```powershell
uv sync --no-config
uv run --no-config python internet_connection_check.py
```

Choose a different HTTPS endpoint or timeout when needed:

```powershell
uv run --no-config python internet_connection_check.py --url https://example.com/ --timeout 5
```

The process exits with code `0` for a successful HTTP response and `1` for a request failure. Importing the module does not make a network request.
