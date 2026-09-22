# PyDoku

A local Tkinter Sudoku game with an interactive grid, puzzle generator, and backtracking solver.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py
```

Click an empty cell and enter a digit from 1 to 9. Use **Solve** to complete the current puzzle, **Generate** to create another puzzle, and the adjacent selector to adjust generation difficulty.

## Notes

The app includes two built-in example puzzles. Generated puzzles use random clue removal and may not have a unique solution. The app uses only the Python standard library, runs locally, and does not create files or make network requests.

The bundled [MIT License](LICENSE) applies to this project.
