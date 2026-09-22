# NASA APOD Viewer

A command-line viewer for NASA's Astronomy Picture of the Day (APOD) API. It prints the title, date, media type, links, copyright notice when present, and a wrapped explanation.

## Requirements

- Python 3.14 or later
- Internet access to `api.nasa.gov` when fetching APOD data

The project has no third-party Python dependencies.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py --help
```

Fetch today's APOD:

```powershell
uv run --no-config python main.py
```

Fetch a specific day or up to ten random entries:

```powershell
uv run --no-config python main.py --date 2024-01-01
uv run --no-config python main.py --random 3
```

## API key

The viewer uses NASA's public `DEMO_KEY` by default, which is rate-limited. For a personal key, set `NASA_API_KEY` in your Windows user environment and start a new terminal before running the command. You can also pass a key for one invocation with `--api-key`; avoid recording that command in shared shell history.

## Limits

- Results are fetched live and are not cached.
- The program displays media links; it does not download images or videos.
- Random results are limited to one through ten entries to keep requests bounded.
