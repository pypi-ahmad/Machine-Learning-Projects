# File Browser GUI

A local Tkinter file browser for navigating folders, previewing small text files, viewing properties, copying paths, opening files with the system default application, and renaming items.

## Requirements

- Python 3.13 or later with Tkinter support
- [uv](https://docs.astral.sh/uv/)

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Start in a specific existing folder:

```powershell
uv run --no-config python main.py "D:\path\to\folder"
```

## Behavior and safety

- The app only reads directory listings and previews local text files up to 200 KB.
- Opening a file delegates to the operating system's default application.
- Renaming accepts a filename only; path separators and `.` or `..` are rejected.
- Permanent deletion is intentionally disabled. This project does not implement a recycle-bin workflow, so removing files would be too easy to get wrong.
- The browser does not upload or transmit file contents.
