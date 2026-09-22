# IP Geolocation

A standard-library CLI for looking up IP metadata, resolving hostnames, and reverse DNS checks.

```powershell
uv sync --no-config
uv run --no-config python main.py 8.8.8.8
```

Without an argument, the interactive menu can also request your public IP. Lookups contact ip-api.com, ipinfo.io, or api.ipify.org; no live request was made during local verification.
