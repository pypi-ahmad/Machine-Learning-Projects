# Text Editor

> A GUI text editor built with Python's tkinter, supporting file open, save, and scrollable text editing.

## Overview

This is a desktop text editor built using tkinter. It provides a text editing area with scrollbar support and a File menu for opening and saving text files. The editor opens with a 600x550 window and includes an `examples/` directory with sample text files.

## Features

- Scrollable text area with padding
- File > Open dialog (defaults to `./examples` directory)
- File > Save dialog
- File > Quit menu option
- Standard text editing (type, select, copy, paste via OS defaults)

## Project Structure

```
Text-Editor/
├── tEditor.py
├── examples/
│   ├── one.txt
│   └── two.txt
└── README.md
```

## Requirements

- Python 3.x
- `tkinter` (included with standard Python installations)

## Installation

```bash
cd "Text-Editor"
```

No additional installation needed — tkinter is included with Python.

## Usage

```bash
python tEditor.py
```

The editor window opens with:
- A text area filling the window
- A scrollbar on the right side
- A **File** menu with **Open**, **Save**, and **Quit** options

**Opening files:** File > Open — defaults to the `./examples` directory. Select any text file to load its contents into the editor.

**Saving files:** File > Save — opens a Save As dialog. Enter a filename and location to save the current text content.

## How It Works

1. Creates a `Tk` root window (600x550 pixels) titled "TextEditor"
2. Builds a `Frame` containing a `Text` widget and a `Scrollbar`
3. The `TextEditor` class provides three methods:
   - `open_file()`: Opens a file dialog, clears the text area, reads the selected file, and inserts its content
   - `save_file()`: Opens a save dialog, gets all text from the area, and writes it to the selected file
   - `quit_app()`: Calls `root.quit()` to close the application
4. A `Menu` bar is configured with File > Open, Save, separator, Quit
5. `root.mainloop()` starts the tkinter event loop

## Configuration

- Default "Open" directory: Hardcoded to `./examples` in the `askopenfilename` call
- Window size: Hardcoded to 600x550 pixels
- Window title: Hardcoded to `"TextEditor"`

## Limitations

- No "New File" option — must manually clear text or reopen
- No "Save" (overwrite) — only "Save As" with a file dialog each time
- No undo/redo functionality
- No syntax highlighting or line numbers
- No keyboard shortcuts (Open/Save/Quit are menu-only)
- No confirmation dialog when closing with unsaved changes
- The `quit_app` method is `@staticmethod` but references the global `root` variable
- Window size is fixed (not responsive to content size)

## Security Notes

No security concerns identified.

## License

Not specified.
