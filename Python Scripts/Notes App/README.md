# Notes App

A local, terminal-based notes application. It stores each note as a UTF-8 text file and supports creating, listing, reading, editing, searching, and deleting notes.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Choose a numbered option from the menu. When entering note content, write `###` on its own line to finish. The same marker finishes edited content.

## Storage and safety

- Notes are stored locally in the `notes` directory beside `main.py`.
- The app does not use a network connection or send note content elsewhere.
- Deleting a note requires a `y` confirmation and permanently removes its text file.
- Note filenames are derived from the title; unsupported filename characters are replaced with underscores.

## Limits

- Notes use plain text only; there is no formatting, encryption, synchronization, or version history.
- Two notes with titles that resolve to the same filename will share one file.
