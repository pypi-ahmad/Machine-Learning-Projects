"""Conway's Game of Life — CLI simulation.

Runs John Conway's Game of Life in the terminal.
Supports random boards, preset patterns (Glider, Blinker, etc.),
custom boards, and an animation loop.

Usage:
    python main.py
"""

import os
import random
import sys
import time


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def empty_grid(rows: int, cols: int) -> list[list[int]]:
    return [[0] * cols for _ in range(rows)]


def random_grid(rows: int, cols: int, density: float = 0.35) -> list[list[int]]:
    return [[1 if random.random() < density else 0 for _ in range(cols)]
            for _ in range(rows)]


def step(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    new = empty_grid(rows, cols)
    for r in range(rows):
        for c in range(cols):
            neighbors = sum(
                grid[(r + dr) % rows][(c + dc) % cols]
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
            )
            if grid[r][c]:
                new[r][c] = 1 if neighbors in (2, 3) else 0
            else:
                new[r][c] = 1 if neighbors == 3 else 0
    return new


def population(grid: list[list[int]]) -> int:
    return sum(sum(row) for row in grid)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

ALIVE = "█"
DEAD  = " "


def render(grid: list[list[int]], gen: int, pop: int) -> str:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    top    = "┌" + "─" * cols + "┐"
    bottom = "└" + "─" * cols + "┘"
    lines  = [f"  Gen {gen:<6}  Population: {pop:<6}"]
    lines.append("  " + top)
    for row in grid:
        lines.append("  │" + "".join(ALIVE if c else DEAD for c in row) + "│")
    lines.append("  " + bottom)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preset patterns
# ---------------------------------------------------------------------------

PATTERNS: dict[str, list[tuple[int, int]]] = {
    "glider":     [(0,1),(1,2),(2,0),(2,1),(2,2)],
    "blinker":    [(5,4),(5,5),(5,6)],
    "toad":       [(5,4),(5,5),(5,6),(6,5),(6,6),(6,7)],
    "beacon":     [(4,4),(4,5),(5,4),(6,7),(7,6),(7,7)],
    "block":      [(5,5),(5,6),(6,5),(6,6)],
    "r-pentomino":[(4,5),(4,6),(5,4),(5,5),(6,5)],
    "gosper":     [  # Gosper glider gun (partial)
        (5,1),(5,2),(6,1),(6,2),
        (5,11),(6,11),(7,11),(4,12),(8,12),(3,13),(9,13),(3,14),(9,14),
        (6,15),(4,16),(8,16),(5,17),(6,17),(7,17),(6,18),
    ],
}


def apply_pattern(grid: list[list[int]], pattern: list[tuple[int, int]],
                   offset_r: int = 0, offset_c: int = 0) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    for r, c in pattern:
        nr, nc = r + offset_r, c + offset_c
        if 0 <= nr < rows and 0 <= nc < cols:
            grid[nr][nc] = 1
    return grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Conway's Game of Life
----------------------
1. Random board
2. Choose a preset pattern
3. Enter custom pattern (RLE or coordinates)
0. Quit
"""


def run_animation(grid: list[list[int]], max_gen: int = 200,
                   delay: float = 0.1) -> None:
    print("  Press Ctrl+C to stop.\n")
    gen = 0
    try:
        while gen < max_gen:
            # Clear screen
            if sys.platform == "win32":
                os.system("cls")
            else:
                print("\033[H\033[J", end="")
            pop = population(grid)
            print(render(grid, gen, pop))
            if pop == 0:
                print("  All cells died.")
                break
            grid = step(grid)
            gen += 1
            time.sleep(delay)
    except KeyboardInterrupt:
        pass
    print(f"\n  Stopped at generation {gen}.")


def main() -> None:
    rows, cols = 24, 60

    print("Conway's Game of Life")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            r_s = input("  Rows (default 24): ").strip()
            c_s = input("  Cols (default 60): ").strip()
            rows = int(r_s) if r_s.isdigit() else 24
            cols = int(c_s) if c_s.isdigit() else 60
            density_s = input("  Density 0-1 (default 0.35): ").strip()
            density = float(density_s) if density_s else 0.35
            grid = random_grid(rows, cols, density)
            gen_s = input("  Max generations (default 200): ").strip()
            max_gen = int(gen_s) if gen_s.isdigit() else 200
            delay_s = input("  Delay seconds (default 0.1): ").strip()
            delay = float(delay_s) if delay_s else 0.1
            run_animation(grid, max_gen, delay)

        elif choice == "2":
            print(f"  Available patterns: {', '.join(PATTERNS)}")
            name = input("  Pattern name: ").strip().lower()
            if name not in PATTERNS:
                print("  Unknown pattern.")
                continue
            grid = empty_grid(rows, cols)
            grid = apply_pattern(grid, PATTERNS[name])
            gen_s = input("  Max generations (default 200): ").strip()
            max_gen = int(gen_s) if gen_s.isdigit() else 200
            run_animation(grid, max_gen, 0.1)

        elif choice == "3":
            grid = empty_grid(rows, cols)
            print("  Enter alive cells as 'row,col' pairs (blank to finish):")
            while True:
                line = input("  > ").strip()
                if not line:
                    break
                try:
                    r, c = map(int, line.split(","))
                    if 0 <= r < rows and 0 <= c < cols:
                        grid[r][c] = 1
                except ValueError:
                    print("  Format: row,col")
            gen_s = input("  Max generations (default 200): ").strip()
            max_gen = int(gen_s) if gen_s.isdigit() else 200
            run_animation(grid, max_gen, 0.1)

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
