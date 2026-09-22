# File Renamer

A preview-first interactive tool for renaming files by text replacement, regular expression, prefix or suffix, numbering, date, case, or extension.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Choose a strategy, then select an existing directory. The tool lists the proposed rename mapping before it changes anything.

## Safety behavior

- Type `RENAME` exactly to apply a previewed set of changes.
- New names cannot contain a path, `.` or `..`.
- Duplicate destinations and unrelated existing files are rejected during preview.
- When an existing source file is also a planned destination, the operation is skipped rather than overwriting that file.
- The project includes `prank/` and `workedOn/` sample image folders. They are not modified unless you select them and explicitly confirm.

The older hardcoded scripts were removed because they targeted another computer's path and performed direct renames without preview or collision protection.
