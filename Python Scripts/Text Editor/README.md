# Text Editor

A small Tkinter desktop editor for plain UTF-8 text files. It includes line numbers, word count, undo and redo, file open and save actions, keyboard shortcuts, and find-and-replace.

## Run it

```powershell
uv sync
uv run python main.py
```

The app opens a desktop window. It is intended for interactive local use on a system with Tkinter available.

## Features

- Open, save, save as, and start a new file
- Line numbers, cursor position, and word count
- Undo, redo, cut, copy, paste, and select all
- Find and replace all matching text
- A confirmation prompt before discarding unsaved work

## Keyboard shortcuts

| Action | Shortcut |
| --- | --- |
| New file | Ctrl+N |
| Open file | Ctrl+O |
| Save | Ctrl+S |
| Save as | Ctrl+Shift+S |
| Select all | Ctrl+A |
| Find and replace | Ctrl+H |

The older `tEditor.py` remains in the folder as a legacy example. Run `main.py` for the maintained editor.

## Dependencies

- Python 3.14+
- Tkinter, included with standard Windows Python installations
