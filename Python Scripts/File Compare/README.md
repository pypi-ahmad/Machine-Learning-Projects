# File Compare

A dependency-free command-line tool that prints non-blank lines shared by two UTF-8 text files.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Compare files

```powershell
uv sync --no-config
uv run --no-config python file.py first.txt second.txt
```

The output preserves the order of `first.txt`, removes duplicate shared lines, and does not modify either input file.

Run without arguments to compare this project's bundled `1.txt` and `2.txt` examples:

```powershell
uv run --no-config python file.py
```

## Write a result file

Writing is opt-in. Choose an output path explicitly:

```powershell
uv run --no-config python file.py first.txt second.txt --output shared.txt
```

Existing output files are protected. Add `--overwrite` only when replacing the destination is intentional.
