# Text File Splitter

A terminal tool for splitting UTF-8 text files by line count, approximate character count, delimiter, or a requested number of roughly equal line-based parts. It can also merge selected text files in sorted filename order.

## Run it

```powershell
uv sync
uv run python main.py
```

Choose an operation and provide the requested paths. Split files are written to the source folder unless you enter a different output directory.

## Notes

- Size splitting uses the decoded text length as an approximate byte limit and prefers a nearby newline.
- Equal splitting creates no more than the requested number of non-empty line chunks.
- Merging overwrites the output path you provide, so choose that path carefully.

## Dependencies

- Python 3.14+
- No third-party packages
