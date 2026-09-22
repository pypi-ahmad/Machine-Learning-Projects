# FLAMES Game

> A Tkinter implementation of the classic FLAMES relationship game, which determines a relationship category from two names.

## Overview

FLAMES compares the letters in two names and uses the number of remaining non-common letters to cycle through "FLAMES" (Friends, Love, Affection, Marriage, Enemy, Siblings). This version provides a Tkinter interface.

## Features

- Tkinter-based GUI with input fields for two names
- Computes the FLAMES result using the classic letter-elimination algorithm
- Displays the relationship status: Friends, Love, Affection, Marriage, Enemy, or Siblings
- Clear button to reset all fields
- Color-coded labels (light green, light blue, bisque)

## Project Structure

```
Flames Game/
├── flames_game_gui.py   # Main GUI application
├── pyproject.toml       # uv project configuration
└── uv.lock              # Locked Python environment
```

## Requirements

- Python 3.13 or later
- `tkinter` (included with standard Python)

## Installation

```bash
cd "Python Scripts/Flames Game"
uv sync --no-config
```

The game has no third-party dependencies.

## Usage

```bash
uv run --no-config python flames_game_gui.py
```

1. Enter the first name in the "Name 1" field.
2. Enter the second name in the "Name 2" field.
3. Click **Flame** to see the relationship status.
4. Click **Clear** to reset all fields.

## How it works

### FLAMES Algorithm (`result_flame` function)

1. Converts both names to character lists (spaces are stripped).
2. Removes common characters from both lists (one occurrence at a time).
3. Counts the total remaining characters across both names.
4. Starts with the list `["Friends", "Love", "Affection", "Marriage", "Enemy", "Siblings"]`.
5. Repeatedly calculates `count % len(result) - 1` as the split index.
6. Removes the element at that index and reorders the list.
7. Continues until only one element remains — that's the result.

### GUI

- Window size: 350×125 pixels
- Background: light pink
- Uses `grid` layout manager for labels and entry fields
- Two buttons: "Flame" (coral) and "Clear" (indian red)

## Configuration

No configuration files. All appearance values are hardcoded.

## Limitations

- The algorithm removes only the first occurrence of each common character — order of removal can vary
- Spaces in names are stripped but other non-alphabetic characters are not handled
- The window is not resizable
- This is a game for entertainment, not a relationship assessment

## Security Notes

No security concerns.

## License

Not specified.
