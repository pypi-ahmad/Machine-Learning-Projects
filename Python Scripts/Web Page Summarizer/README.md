# Web page summarizer

`app.py` fetches HTML pages and creates short extractive summaries. It can handle
one URL or a CSV that contains a URL column. The project uses only Python's
standard library, so `uv sync` installs no third-party packages.

## Install

```powershell
cd "Python Scripts/Web Page Summarizer"
uv sync
```

## Summarize one page

```powershell
uv run python app.py url https://example.com --sentences 2
```

Only absolute `http` and `https` URLs are accepted. The request timeout is 10
seconds and the response limit is 2,000,000 bytes by default. Adjust either
value when needed:

```powershell
uv run python app.py url https://example.com --sentences 3 --timeout 20 --max-bytes 500000
```

The summary selects sentences using word frequency and preserves their original
page order. It is extractive, so it does not generate new wording.

## Summarize a CSV

The input CSV must have a `website` column by default.

```powershell
uv run python app.py csv urls.csv summaries.csv --sentences 2
```

Use `--column` when URLs are stored under another column name:

```powershell
uv run python app.py csv urls.csv summaries.csv --column url
```

The output is always a new CSV. It retains the input columns and adds `summary`
and `error`. A failed URL leaves its summary blank and records the error, so the
rest of the CSV can still finish.

## Limits

The tool reads server-rendered HTML only. Pages that require JavaScript, login,
or bot verification may not provide useful text. It ignores text inside
`script`, `style`, and `noscript` elements, but it is not a full article parser.

Run `uv run python app.py --help` for the command reference.
