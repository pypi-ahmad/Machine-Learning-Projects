# Wikipedia infobox scraper

`main.py` is a Tkinter desktop tool that fetches the infobox of an English
Wikipedia page and displays its available fields.

## Install

```powershell
cd "Python Scripts/Wikipedia Infobox Scraper"
uv sync
uv run python main.py
```

## Use

1. Enter a page title, such as `Python programming language`.
2. Select **Fetch infobox**.
3. Review the page URL and extracted fields in the window.

The request runs outside the Tkinter event loop, so the interface stays usable
while Wikipedia responds. Requests use a 15-second timeout and a descriptive
user agent.

## Parsing behavior

Titles are normalized to spaces and encoded into an English Wikipedia page URL.
The tool extracts direct header/value pairs from the first `table.infobox`
element. It reports a clear error when the page cannot be fetched, has no
infobox, or has no readable fields.

## Limits

Only English Wikipedia is queried. The tool does not resolve search results,
disambiguation pages, pages that lack an infobox, or data stored outside the
page's rendered infobox table.
