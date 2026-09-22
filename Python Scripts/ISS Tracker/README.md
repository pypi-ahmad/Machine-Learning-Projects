# ISS Tracker

A standard-library CLI that retrieves the International Space Station's current reported position, people in space, and location-based details.

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Use `--location "New York"` to geocode a city before calculating distance, or `--watch` for repeated live updates. These commands contact public web APIs; no live request was made during local verification.
