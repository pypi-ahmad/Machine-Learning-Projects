# Tic Tac Toe

> A terminal-based two-player Tic Tac Toe game using coordinate input.

## Overview

A command-line Tic Tac Toe game where two players (`x` and `o`) take turns entering row and column coordinates on a 3×3 board. The game checks for wins across rows, columns, and diagonals, and ends when a player wins or the board is full.

## Features

- Two-player gameplay (`x` vs `o`) with alternating turns
- Coordinate-based input system (row + column as a two-digit number, e.g., `11` for row 1, column 1)
- Input validation for out-of-range coordinates and occupied cells
- Win detection for all 8 possible winning combinations (3 rows, 3 columns, 2 diagonals)
- Draw detection when the board is full
- Visual board display after each turn

## Project Structure

```
Tic_tac_toe/
├── tic_tac_toe.py
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only built-in features)

## Installation

```bash
cd "Tic_tac_toe"
```

No package installation required.

## Usage

```bash
python tic_tac_toe.py
```

When prompted, enter a two-digit number where the first digit is the row (1–3) and the second is the column (1–3):

```
enter x and y: 12
```

This places the current player's mark in row 1, column 2.

## How It Works

1. The board is represented as a 3×3 list of lists initialized with empty strings.
2. Players alternate turns — `x` goes first, then `o`.
3. On each turn, the player enters a two-digit coordinate (e.g., `23` = row 2, column 3).
4. `check_xy()` validates the input is a two-digit number within range 1–3.
5. `set_room_state()` checks if the cell is empty and places the mark.
6. `check_for_win()` checks all 8 winning lines (rows, columns, diagonals).
7. `have_empty_room()` checks for a draw condition (no empty cells left).

## Configuration

No configuration needed. All settings are hardcoded.

## Limitations

- No AI opponent — requires two human players.
- Input must be an integer; non-numeric input will cause a crash (`int()` call without error handling).
- The first turn always goes to `x` (the `turn` variable starts as `o` but is swapped before the first move).
- Board uses 1-indexed coordinates, which may be unintuitive.
- Bug: In `check_for_win()`, the vertical check for column 3 references `board[0][0]` instead of `board[0][2]` for the winner name.

## License

Not specified.
