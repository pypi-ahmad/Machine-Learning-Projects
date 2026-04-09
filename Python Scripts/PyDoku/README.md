# PyDoku

> A Sudoku puzzle game with an interactive GUI, puzzle generator, and backtracking solver built with Python's tkinter.

## Overview

PyDoku is a fully interactive Sudoku application that lets users play Sudoku puzzles, generate new random puzzles at varying difficulty levels, and automatically solve any puzzle using a recursive backtracking algorithm. The entire interface is rendered on a tkinter Canvas with clickable cells and inline editing.

## Features

- Interactive 9×9 Sudoku grid rendered on a tkinter Canvas
- Click-to-edit cells with inline Entry widget for number input
- Input validation (digits 0–9 accepted; entering 0 clears the cell)
- Visual feedback: green highlight for editable cells, red for locked clue cells
- Automatic puzzle solver using recursive backtracking
- Random puzzle generator with 5 difficulty levels
- Non-editable clue cells displayed in bold font
- Recursion counter printed to console after solving
- Two built-in example puzzles (including an "extreme" puzzle)

## Project Structure

```
PyDoku/
├── main.py        # Complete application: GUI, solver, generator
├── LICENSE         # MIT License
└── README.md
```

## Requirements

- Python 3.x
- `tkinter` (included with standard Python installations)
- No external dependencies

## Installation

```bash
cd "PyDoku"
python main.py
```

No additional packages need to be installed.

## Usage

Run the application:

```bash
python main.py
```

- **Click a cell** to edit it (editable cells flash green, clue cells flash red)
- **Type a digit (1–9)** to fill in the selected cell
- **Solve button** — automatically solves the current puzzle using backtracking
- **Generate button** — generates a new random puzzle
- **Difficulty selector (1–5)** — dropdown next to Generate; higher values remove more numbers, making the puzzle harder

## How It Works

1. **Grid Rendering**: A 300×300 tkinter Canvas draws a 9×9 grid with thin grey lines for cells and thick black lines for 3×3 box boundaries.
2. **Cell Storage**: Each cell is stored in a dictionary keyed by `(x, y)` with value `[number, editable, canvas_item_id]`.
3. **Editing**: Clicking a cell creates a small Entry widget overlaid on that cell. On key release, the input is validated and the cell is updated.
4. **Solver**: Uses recursive backtracking — finds the first empty cell, tries digits 1–9, checks sub-grid and row/column constraints, recurses, and backtracks if no valid digit is found. The canvas updates visually during solving.
5. **Generator**: Creates three random diagonal 3×3 sub-grids, solves the rest, then removes numbers based on the difficulty level (higher difficulty = more cells removed).
6. **Constraint Checking**: `is_SubGrid_Safe()` checks the 3×3 box rule; `is_Cell_Safe()` checks the row and column rules.

## Configuration

- `canvas_bg`, `line_normal`, `line_thick`, `hbox_green`, `hbox_red` — color constants defined as class attributes in the `Sudoku` class
- `self.canvas_width` / `self.canvas_height` — grid dimensions (default: 300×300)
- `ex1` and `ex2` — hardcoded example puzzles at the bottom of `main.py`; the app starts with `ex1` by default

## Limitations

- The solver updates the canvas synchronously, which can briefly freeze the UI on hard puzzles
- Difficulty generation uses a simple random removal heuristic rather than ensuring a unique solution
- The window is non-resizable (fixed 300×300 grid)
- Only `ex1` is loaded on startup; `ex2` is defined but unused unless manually changed in code
- No save/load functionality for puzzles
- No timer or scoring system

## Security Notes

No security concerns identified.

## License

MIT License (Copyright (c) 2020 Akshit Khajuria)
