# Website snapshot

`snapshot_of_given_website.py` saves a full-page PNG snapshot of an HTTP(S)
website using headless Chrome and Selenium.

## Requirements

- Python 3.13 or later
- Google Chrome installed

Selenium Manager obtains a compatible ChromeDriver when needed. You do not need
to download or configure a separate driver binary.

## Install

```powershell
cd "Python Scripts/Website Snapshot"
uv sync
```

## Usage

```powershell
uv run python snapshot_of_given_website.py https://example.com --output example.png
```

The default output path is `screenshot.png`. The command refuses to replace an
existing file. Use another `--output` path for a new capture.

Set the page-load timeout when a site needs more or less time:

```powershell
uv run python snapshot_of_given_website.py https://example.com --output example.png --timeout 45
```

The tool accepts only absolute `http` and `https` URLs. It uses Chrome DevTools
to capture beyond the visible viewport after the page-load event completes.

## Limits

Pages that require authentication, user interaction, bot verification, or
content loaded after the page-load event may produce incomplete snapshots. The
script does not bypass those restrictions.

Run `uv run python snapshot_of_given_website.py --help` for the command
reference.
