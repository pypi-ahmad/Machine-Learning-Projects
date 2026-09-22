# URL Parser

A terminal tool for inspecting URL components, building URLs, encoding and decoding query strings, URL-encoding text, and resolving relative URLs.

## Run it

```powershell
uv sync
uv run python main.py
```

Choose an action from the interactive menu. URL parsing and conversion happen locally; the tool does not contact the URL you enter.

## Notes

Parsed URLs and query strings are printed to the terminal. Avoid entering URLs that contain passwords, tokens, API keys, or other secrets.

## Dependencies

- Python 3.14+
- No third-party packages
