# Sticky Notes

A local Tkinter desktop app for creating, moving, recoloring, and editing
floating notes.

## Run

```powershell
uv run python main.py
```

Use the control window to create a note or quit the app. Drag a note by its
top bar, edit its text directly, right-click its text area for color and delete
actions, and use the buttons in the note bar to add or remove notes.

## Data storage

Notes are stored in `sticky_notes.json` beside `main.py`. The file is created
when notes are saved and persists their text, color, position, and current
window size between sessions. Deleting a note removes it from that file.

This is a local single-user utility. It does not synchronize notes, encrypt
their contents, or protect them from another user with access to the computer.
Avoid storing secrets or sensitive information in its notes.

## Dependencies

The project uses only Python's standard library, including Tkinter. uv records
the Python requirement and provides the reproducible environment.
