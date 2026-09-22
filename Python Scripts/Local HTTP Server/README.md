# Local HTTP Server

A standard-library development server for a local directory, with optional CORS, basic authentication, cache controls, and request statistics.

```powershell
uv sync --no-config
uv run --no-config python main.py . --port 8000
```

The server binds to `127.0.0.1` only, so it is not exposed to other devices by default. Do not serve sensitive files or use basic-auth credentials on untrusted networks.
