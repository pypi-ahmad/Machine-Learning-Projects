# Wikipedia Scraper

Search and retrieve Wikipedia content from the command line. The default command prints a short article summary; larger outputs are explicit options.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

## Install

```powershell
cd "Python Scripts\Wikipedia Scraper"
uv sync
```

## Usage

Print a three-sentence English summary:

```powershell
uv run python ".\Wikipedia Scrapper in Python.py" Python
```

Search instead of retrieving a summary:

```powershell
uv run python ".\Wikipedia Scrapper in Python.py" Python --search
```

Use a different Wikipedia language edition:

```powershell
uv run python ".\Wikipedia Scrapper in Python.py" Python --language fr
```

Full content, links, and image URLs are opt-in:

```powershell
uv run python ".\Wikipedia Scrapper in Python.py" Python --full
uv run python ".\Wikipedia Scrapper in Python.py" Python --links
uv run python ".\Wikipedia Scrapper in Python.py" Python --images
```

Preview a request without using the network:

```powershell
uv run python ".\Wikipedia Scrapper in Python.py" Python --dry-run
```

## Notes

- Wikipedia content and availability depend on the selected language edition and network access.
- Ambiguous topics produce a list of candidate titles.
- Respect Wikipedia's terms and the licenses that apply to retrieved content.
