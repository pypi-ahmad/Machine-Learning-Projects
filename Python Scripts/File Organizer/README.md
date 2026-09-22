# File Organizer

A preview-first command-line tool that organizes one folder by file extension or modification date.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Preview by extension

```powershell
uv sync --no-config
uv run --no-config python main.py "C:\path\to\folder"
```

The default is a preview. It prints every proposed source-to-destination mapping and does not create folders or move files.

## Apply a reviewed plan

```powershell
uv run --no-config python main.py "C:\path\to\folder" --apply
```

The command asks you to type `MOVE` before changing files. Add `--copy` to copy instead, then type `COPY` when prompted.

## Other modes

```powershell
uv run --no-config python main.py "C:\source" --destination "C:\organized" --strategy date --apply
uv run --no-config python main.py "C:\source" --destination "C:\organized" --recursive
```

Date folders use `%Y/%m` by default. Recursive organization requires a destination outside the source folder so newly created output folders are not processed again.

## Safety

- Existing destination files are never overwritten; name collisions receive a numeric suffix.
- Destination folders are created only after explicit confirmation.
- Review the full preview before applying a move. Moves can be difficult to undo if another process modifies files at the same time.
