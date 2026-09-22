# Unique Words in File

A terminal tool that finds words occurring exactly once in a UTF-8 text file, comparing words case-insensitively and printing the result in alphabetical order.

## Run it

```powershell
uv sync
uv run python unique.py
```

Without an argument, the script analyzes the bundled `text_file.txt` sample. To inspect another file:

```powershell
uv run python unique.py path\to\document.txt
```

Words are extracted with the pattern `\w+`, which includes letters, digits, and underscores. Files are read as UTF-8; undecodable characters are replaced for analysis.

## Dependencies

- Python 3.14+
- No third-party packages
