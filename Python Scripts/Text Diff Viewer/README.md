# Text Diff Viewer

A terminal tool for comparing two text blocks or two UTF-8 text files. It can show a unified diff, a side-by-side comparison, or only a similarity ratio.

## Run it

```powershell
uv sync
uv run python main.py
```

For text input, finish each block by entering `###` on its own line. For file comparison, provide the paths when prompted. Files are read as UTF-8 and undecodable characters are replaced for display.

## Dependencies

- Python 3.14+
- No third-party packages
