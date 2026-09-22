# Open Port Scanner

A command-line utility that checks one inclusive TCP port range on a single host.

## Authorization

Only scan hosts and networks you own or have explicit permission to test. Port scans can be logged, rate-limited, or treated as unauthorized activity.

## Run

```powershell
uv sync --no-config
uv run --no-config python fetch_open_port.py --help
uv run --no-config python fetch_open_port.py 127.0.0.1 --ports 50:500
```

Use `--timeout` to set the per-port connection timeout in seconds:

```powershell
uv run --no-config python fetch_open_port.py 127.0.0.1 --ports 8000:8010 --timeout 0.2
```

## Behavior and limits

- The scanner resolves one hostname or IPv4 address and checks TCP connections sequentially.
- It reports only ports that accept the connection attempt; it does not identify services, collect banners, or scan UDP.
- The default range is ports 50 through 500, with a 0.5-second timeout per port.
- Network policy, firewalls, and packet loss can make port state appear closed or filtered.
