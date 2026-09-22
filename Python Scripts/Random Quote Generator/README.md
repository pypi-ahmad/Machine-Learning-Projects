# Random Quote Generator

Fetch a random quote from Quotable when available, with a built-in fallback quote bank.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py
uv run --no-config python main.py --author "Einstein"
```

Use the interactive menu to get a new quote or save and manage favourites. Saved favourites are written to `quote_favourites.json` beside `main.py`.

## Notes

Live quotes require internet access to Quotable. If the request fails, the interactive mode uses the bundled fallback quotes. Treat quote attribution as service-provided data and verify it before reuse.
