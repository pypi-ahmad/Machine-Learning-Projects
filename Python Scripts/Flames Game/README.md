# FLAMES Game

> A Tkinter GUI implementation of the classic FLAMES relationship game that determines the relationship between two people based on their names.

## Overview

FLAMES is a fun game where the letters in two names are compared, and the count of remaining (non-common) letters is used to cycle through the word "FLAMES" (Friends, Love, Affection, Marriage, Enemy, Siblings) to determine a relationship status. This implementation provides a graphical interface using Tkinter.

## Features

- Tkinter-based GUI with input fields for two names
- Computes the FLAMES result using the classic letter-elimination algorithm
- Displays the relationship status: Friends, Love, Affection, Marriage, Enemy, or Siblings
- Clear button to reset all fields
- Color-coded labels (light green, light blue, bisque)

## Project Structure

```
Flames-Game/
└── flames_game_gui.py   # Main GUI application
```

## Requirements

- Python 3.x
- `tkinter` (included with standard Python)

## Installation

```bash
cd "Flames-Game"
```

No package installation needed.

## Usage

```bash
python flames_game_gui.py
```

1. Enter the first name in the "Name 1" field.
2. Enter the second name in the "Name 2" field.
3. Click **Flame** to see the relationship status.
4. Click **Clear** to reset all fields.

## How It Works

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
- No input validation (empty names will produce a result based on 0 remaining characters)
- The window is not resizable
- The result is inserted into the Status field at position 10, which may cause display issues for repeated clicks without clearing

## Security Notes

No security concerns.

## License

Not specified.
