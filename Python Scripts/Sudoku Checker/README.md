# Sudoku Checker

A small command-line tool for checking a 9×9 Sudoku grid, highlighting conflicts, and solving valid puzzles with backtracking.

## Run it

```powershell
uv sync
uv run python main.py
```

The interactive prompt starts with a sample puzzle. Use `show`, `check`, `solve`, `load`, `edit`, `reset`, or `quit`.

## Load a puzzle from a file

Provide a file containing exactly 81 digits or dots. Use `0` or `.` for an empty cell.

```powershell
uv run python main.py --file puzzle.txt
uv run python main.py --file puzzle.txt --solve
```

`--solve` without a file solves the built-in sample puzzle.

## Dependencies

- Python 3.14+
- No third-party packages
