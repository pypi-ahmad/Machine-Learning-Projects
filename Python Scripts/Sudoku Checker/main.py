"""Sudoku Checker — CLI tool.

Input a 9×9 Sudoku puzzle, check its validity,
highlight errors, and attempt to solve it.

Usage:
    python main.py
    python main.py --file puzzle.txt
    python main.py --solve
"""

import argparse
import copy
import sys
from pathlib import Path


SAMPLE_PUZZLE = [
    [5,3,0, 0,7,0, 0,0,0],
    [6,0,0, 1,9,5, 0,0,0],
    [0,9,8, 0,0,0, 0,6,0],
    [8,0,0, 0,6,0, 0,0,3],
    [4,0,0, 8,0,3, 0,0,1],
    [7,0,0, 0,2,0, 0,0,6],
    [0,6,0, 0,0,0, 2,8,0],
    [0,0,0, 4,1,9, 0,0,5],
    [0,0,0, 0,8,0, 0,7,9],
]


def print_board(board: list[list[int]], errors: set = None) -> None:
    errors = errors or set()
    print()
    for r in range(9):
        if r in (3, 6):
            print("  ├───────┼───────┼───────┤")
        row_str = "  │"
        for c in range(9):
            v = board[r][c]
            s = str(v) if v else "·"
            if (r, c) in errors:
                s = f"\033[91m{s}\033[0m"    # red
            row_str += f" {s}"
            if c in (2, 5):
                row_str += " │"
        row_str += " │"
        print(row_str)
    print()


def check_valid(board: list[list[int]]) -> tuple[bool, set]:
    """Return (is_valid, set_of_error_cells)."""
    errors = set()

    def check_group(cells, coords):
        seen = {}
        for idx, (r, c) in enumerate(coords):
            v = board[r][c]
            if v == 0: continue
            if v in seen:
                errors.add((r, c))
                errors.add(seen[v])
            else:
                seen[v] = (r, c)

    for i in range(9):
        check_group(None, [(i, c) for c in range(9)])          # rows
        check_group(None, [(r, i) for r in range(9)])          # cols
        br, bc = (i // 3) * 3, (i % 3) * 3
        check_group(None, [(br+r, bc+c) for r in range(3) for c in range(3)])  # boxes

    return len(errors) == 0, errors


def is_complete(board: list[list[int]]) -> bool:
    return all(board[r][c] != 0 for r in range(9) for c in range(9))


def candidates(board: list[list[int]], r: int, c: int) -> set[int]:
    used = set()
    for i in range(9):
        used.add(board[r][i])
        used.add(board[i][c])
    br, bc = (r // 3) * 3, (c // 3) * 3
    for dr in range(3):
        for dc in range(3):
            used.add(board[br+dr][bc+dc])
    return set(range(1, 10)) - used


def solve(board: list[list[int]]) -> bool:
    """Backtracking solver. Modifies board in place. Returns True if solved."""
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for v in candidates(board, r, c):
                    board[r][c] = v
                    if solve(board):
                        return True
                    board[r][c] = 0
                return False
    return True


def parse_board_string(text: str) -> list[list[int]]:
    """Parse a 81-digit string or 9-line format into a 9×9 grid."""
    digits = [c for c in text if c.isdigit() or c in ".0"]
    if len(digits) == 81:
        row = []
        board = []
        for i, ch in enumerate(digits):
            row.append(0 if ch in ".0" else int(ch))
            if len(row) == 9:
                board.append(row)
                row = []
        return board
    raise ValueError("Expected exactly 81 digits/dots.")


def interactive() -> None:
    board = [row[:] for row in SAMPLE_PUZZLE]
    print("=== Sudoku Checker ===")
    print("Commands: show | check | solve | load | edit | reset | quit\n")

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"):
            break
        elif cmd == "show":
            valid, errs = check_valid(board)
            print_board(board, errs)
            print(f"  Valid: {valid}  |  Filled: {sum(board[r][c]!=0 for r in range(9) for c in range(9))}/81")
        elif cmd == "check":
            valid, errs = check_valid(board)
            if valid and is_complete(board):
                print("  ✅ Puzzle is correctly solved!")
            elif valid:
                print(f"  ✅ No errors so far (puzzle not yet complete).")
            else:
                print_board(board, errs)
                print(f"  ❌ {len(errs)} conflicting cell(s) highlighted in red.")
        elif cmd == "solve":
            b = copy.deepcopy(board)
            valid, _ = check_valid(b)
            if not valid:
                print("  ❌ Fix errors before solving.")
                continue
            if solve(b):
                print("  ✅ Solved!")
                print_board(b)
                board = b
            else:
                print("  ❌ No solution found for this puzzle.")
        elif cmd == "load":
            print("  Paste 81 digits (0 or . for empty):")
            try:
                text = input("  > ").strip()
                board = parse_board_string(text)
                print_board(board)
            except Exception as e:
                print(f"  Error: {e}")
        elif cmd == "edit":
            print("  Enter: row col value  (e.g. 1 5 3; use 0 to clear)")
            try:
                r, c, v = map(int, input("  > ").split())
                if not (1<=r<=9 and 1<=c<=9 and 0<=v<=9):
                    print("  Out of range.")
                else:
                    board[r-1][c-1] = v
                    print_board(board)
            except Exception as e:
                print(f"  Error: {e}")
        elif cmd == "reset":
            board = [row[:] for row in SAMPLE_PUZZLE]
            print("  Board reset to sample puzzle.")
        else:
            print("  Commands: show | check | solve | load | edit | reset | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Sudoku Checker & Solver")
    parser.add_argument("--file",  metavar="PATH", help="Load puzzle from text file")
    parser.add_argument("--solve", action="store_true", help="Auto-solve on load")
    args = parser.parse_args()

    board = [row[:] for row in SAMPLE_PUZZLE]

    if args.file:
        try:
            board = parse_board_string(Path(args.file).read_text())
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr); sys.exit(1)

    if args.solve or args.file:
        print("  Puzzle:")
        print_board(board)
        valid, errs = check_valid(board)
        if not valid:
            print_board(board, errs)
            print(f"❌ Puzzle has {len(errs)} conflicting cells.")
            sys.exit(1)
        b = copy.deepcopy(board)
        if solve(b):
            print("  Solution:")
            print_board(b)
        else:
            print("  No solution found.")
        return

    interactive()


if __name__ == "__main__":
    main()
