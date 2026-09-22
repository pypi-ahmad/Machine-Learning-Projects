# Public Holidays Checker

Look up public holidays for a country and year through the Nager.Date API.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py --country US --year 2026
uv run --no-config python main.py --country GB --next
uv run --no-config python main.py --list-countries
```

Run without arguments for interactive commands.

## Notes

The app fetches live holiday data from the Nager.Date API and needs an internet connection. No API key is required. Holiday availability and naming are provided by that service, so check official local sources for decisions with legal, employment, or travel consequences.
