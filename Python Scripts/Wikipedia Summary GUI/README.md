# Wikipedia summary GUI

`summary.py` is a Tkinter desktop tool that displays the introductory plaintext
summary of an English Wikipedia article.

## Install

```powershell
cd "Python Scripts/Wikipedia Summary GUI"
uv sync
uv run python summary.py
```

## Use

1. Enter an article title, such as `Python programming language`.
2. Select **Get summary**.
3. Read the resolved page title and introductory summary in the text area.

The request runs outside the Tkinter event loop, so the window remains
responsive while Wikipedia responds. Requests use a 15-second timeout and a
descriptive user agent.

## API behavior

The application calls the English Wikipedia MediaWiki action API for a
plaintext introductory extract. Redirects are resolved by the API, and empty,
missing, or unavailable extracts are reported as errors.

## Limits

Only English Wikipedia is queried. The tool does not disambiguate ambiguous
search terms, return full articles, or work offline. Network failures and API
responses without an introductory extract are shown in the interface.
