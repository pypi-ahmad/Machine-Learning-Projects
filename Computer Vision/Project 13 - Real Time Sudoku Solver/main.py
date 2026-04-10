"""Entry point for Real Time Sudoku Solver (Project 13).

Launches the Sudoku solver module.
Usage:
    python main.py
"""

import runpy
import sys
from pathlib import Path

# Ensure project directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "Sudoku.py"),
        run_name="__main__",
    )
