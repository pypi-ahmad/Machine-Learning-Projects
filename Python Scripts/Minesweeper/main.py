"""Minesweeper — CLI game.

Classic Minesweeper with beginner/intermediate/expert modes.
Type 'F row col' to flag/unflag a cell, 'row col' to reveal it.

Usage:
    python main.py
"""

import random
import time


# ---------------------------------------------------------------------------
# Game logic
# ---------------------------------------------------------------------------

MINE = -1


def create_board(rows: int, cols: int, mines: int,
                  safe_r: int, safe_c: int) -> list[list[int]]:
    """Create board ensuring (safe_r, safe_c) and its neighbours are mine-free."""
    safe_zone = {(safe_r + dr, safe_c + dc)
                 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                 if 0 <= safe_r + dr < rows and 0 <= safe_c + dc < cols}
    positions = [(r, c) for r in range(rows) for c in range(cols)
                  if (r, c) not in safe_zone]
    mine_positions = set(random.sample(positions, min(mines, len(positions))))

    board = [[0] * cols for _ in range(rows)]
    for r, c in mine_positions:
        board[r][c] = MINE

    # Count adjacent mines
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == MINE:
                continue
            count = sum(
                1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
                and 0 <= r + dr < rows and 0 <= c + dc < cols
                and board[r + dr][c + dc] == MINE
            )
            board[r][c] = count
    return board


def reveal(board, visible, flagged, r, c, rows, cols):
    """Flood-fill reveal from (r, c)."""
    if visible[r][c] or flagged[r][c]:
        return
    visible[r][c] = True
    if board[r][c] == 0:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    reveal(board, visible, flagged, nr, nc, rows, cols)


def check_win(board, visible, rows, cols, mines):
    revealed = sum(visible[r][c] for r in range(rows) for c in range(cols))
    return revealed == rows * cols - mines


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

COLORS = {
    1: "\033[34m", 2: "\033[32m", 3: "\033[31m", 4: "\033[35m",
    5: "\033[33m", 6: "\033[36m", 7: "\033[37m", 8: "\033[90m",
}
RESET = "\033[0m"


def display(board, visible, flagged, rows, cols, reveal_all=False):
    # Column header
    header = "    " + "  ".join(f"{c:>2}" for c in range(cols))
    print("\n" + header)
    print("    " + "──" * cols)
    for r in range(rows):
        row_str = f"{r:>3} │"
        for c in range(cols):
            if reveal_all and board[r][c] == MINE:
                row_str += " 💣"
            elif flagged[r][c]:
                row_str += " 🚩"
            elif not visible[r][c]:
                row_str += "  ■"
            elif board[r][c] == MINE:
                row_str += " 💥"
            elif board[r][c] == 0:
                row_str += "  ·"
            else:
                n = board[r][c]
                color = COLORS.get(n, "")
                row_str += f" {color}{n}{RESET} "
        print(row_str)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PRESETS = {
    "b": (9, 9, 10,    "Beginner"),
    "i": (16, 16, 40,  "Intermediate"),
    "e": (16, 30, 99,  "Expert"),
}


def parse_command(cmd: str, rows: int, cols: int):
    """Parse 'r c' or 'f r c'. Returns (action, row, col) or None."""
    parts = cmd.strip().lower().split()
    try:
        if len(parts) == 2:
            r, c = int(parts[0]), int(parts[1])
            if 0 <= r < rows and 0 <= c < cols:
                return "reveal", r, c
        elif len(parts) == 3 and parts[0] == "f":
            r, c = int(parts[1]), int(parts[2])
            if 0 <= r < rows and 0 <= c < cols:
                return "flag", r, c
    except ValueError:
        pass
    return None


def main() -> None:
    print("💣 Minesweeper")
    print("  Commands: 'row col' to reveal, 'f row col' to flag")

    while True:
        print("\n  Difficulty: (b)eginner / (i)ntermediate / (e)xpert / (c)ustom")
        diff = input("  Choice (default b): ").strip().lower() or "b"

        if diff in PRESETS:
            rows, cols, mines, name = PRESETS[diff]
            print(f"  {name}: {rows}×{cols}, {mines} mines")
        elif diff == "c":
            rows  = int(input("  Rows  : ").strip() or "10")
            cols  = int(input("  Cols  : ").strip() or "10")
            mines = int(input("  Mines : ").strip() or "15")
            name  = "Custom"
        else:
            rows, cols, mines, name = PRESETS["b"]

        board   = None
        visible = [[False]*cols for _ in range(rows)]
        flagged = [[False]*cols for _ in range(rows)]
        started = False
        start_t = None

        while True:
            remaining = mines - sum(flagged[r][c] for r in range(rows) for c in range(cols))
            display(board if board else [[0]*cols for _ in range(rows)],
                    visible, flagged, rows, cols)
            if started and start_t:
                elapsed = int(time.time() - start_t)
                print(f"  🚩 Flags remaining: {remaining}   ⏱ {elapsed}s")
            else:
                print(f"  🚩 Flags remaining: {remaining}")

            cmd = input("  > ").strip()
            if cmd.lower() in ("q", "quit"):
                print("  Quitting...")
                return

            parsed = parse_command(cmd, rows, cols)
            if not parsed:
                print("  Invalid command. Format: 'row col' or 'f row col'")
                continue

            action, r, c = parsed

            if action == "flag":
                if visible[r][c]:
                    print("  Cell already revealed.")
                    continue
                flagged[r][c] = not flagged[r][c]
                continue

            # Reveal
            if flagged[r][c]:
                print("  Cell is flagged. Unflag first with 'f row col'.")
                continue

            if not started:
                board = create_board(rows, cols, mines, r, c)
                started = True
                start_t = time.time()

            if board[r][c] == MINE:
                display(board, visible, flagged, rows, cols, reveal_all=True)
                print("  💥 BOOM! You hit a mine. Game over.")
                break

            reveal(board, visible, flagged, r, c, rows, cols)

            if check_win(board, visible, rows, cols, mines):
                elapsed = int(time.time() - start_t)
                display(board, visible, flagged, rows, cols)
                print(f"  🎉 You win! Time: {elapsed}s")
                break

        again = input("\n  Play again? (y/n): ").strip().lower()
        if again != "y":
            print("Bye!")
            break


if __name__ == "__main__":
    main()
