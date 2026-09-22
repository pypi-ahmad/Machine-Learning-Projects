# Live Cricket Score

Windows desktop notifier for score changes shown on Cricbuzz's public live-scores page.

## Setup

Requirements: Windows, Python 3.13+, and a working Windows notification service.

```powershell
cd "Live Cricket Score"
uv sync
```

`uv sync` also installs the compatibility version of `setuptools` required by the legacy `win10toast` package.

## Usage

Start continuous monitoring with a 60-second polling interval:

```powershell
uv run python live_score.py
```

Useful options:

- `--interval SECONDS`: Polling interval. Default: `60`.
- `--once`: Fetch one update and exit.
- `--dry-run`: Print changed scores instead of creating Windows notifications.

Example, one visible console-only check:

```powershell
uv run python live_score.py --once --dry-run
```

Stop continuous monitoring with `Ctrl+C`.

## Behavior

1. Fetches Cricbuzz over HTTPS with a 20-second timeout.
2. Extracts match headings and score text from the current live-scores page.
3. Sends a notification only when a score is new or has changed since the preceding poll.
4. Prints an error if the page cannot be fetched, then retries on the next polling interval.

The source page can change its markup or limit automated requests. This tool does not bypass access controls, rate limits, or CAPTCHA checks. An empty result can mean that no live scores are currently exposed by the page.

## Project files

```text
Live Cricket Score/
├── ipl.ico
├── live_score.py
├── pyproject.toml
└── uv.lock
```
