"""Maze Generator — CLI tool.

Generate random mazes using recursive backtracking (DFS).
Solve with BFS/DFS, display ASCII art, save to file.

Usage:
    python main.py
    python main.py --width 21 --height 11
    python main.py --width 41 --height 21 --solve --save maze.txt
"""

import argparse
import random
import sys
from collections import deque
from pathlib import Path


WALL  = "██"
OPEN  = "  "
START = "🟢"
END   = "🔴"
PATH  = "· "
CRUMB = "◆ "


def generate_maze(width: int, height: int, seed: int = None) -> list[list[str]]:
    """
    Recursive backtracking DFS maze generation.
    width, height must be odd for proper walls.
    Returns a 2D grid of WALL/OPEN strings.
    """
    rng   = random.Random(seed)
    grid  = [[WALL] * width for _ in range(height)]

    def carve(x: int, y: int):
        grid[y][x] = OPEN
        directions = [(2,0),(-2,0),(0,2),(0,-2)]
        rng.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and grid[ny][nx] == WALL:
                grid[y + dy//2][x + dx//2] = OPEN
                carve(nx, ny)

    # Start at (1,1)
    carve(1, 1)
    # Set entrance / exit
    grid[1][0]            = OPEN
    grid[height-2][width-1] = OPEN
    return grid


def solve_maze(grid: list[list[str]], width: int, height: int) -> list[tuple[int,int]] | None:
    """BFS from start (1,0) to end (height-2, width-1)."""
    start = (0, 1)
    end   = (width - 1, height - 2)
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == end:
            return path
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx,ny) not in visited:
                if grid[ny][nx] != WALL:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
    return None


def render(grid: list[list[str]], path: list[tuple[int,int]] = None,
           width: int = 0, height: int = 0) -> str:
    path_set = set(path) if path else set()
    lines    = []
    for y, row in enumerate(grid):
        line = ""
        for x, cell in enumerate(row):
            if   (x, y) == (0, 1):
                line += START
            elif (x, y) == (width-1, height-2):
                line += END
            elif (x, y) in path_set and cell == OPEN:
                line += CRUMB
            else:
                line += cell
        lines.append(line)
    return "\n".join(lines)


def interactive():
    print("=== Maze Generator ===")
    print("Commands: generate | solve | save | quit\n")
    grid   = None
    width  = height = 0
    path   = None

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"):
            break
        elif cmd == "generate":
            w = int(input("  Width  (odd, ≥5) [21]: ").strip() or 21)
            h = int(input("  Height (odd, ≥5) [11]: ").strip() or 11)
            w = w | 1   # ensure odd
            h = h | 1
            width, height = w, h
            grid  = generate_maze(w, h)
            path  = None
            print(render(grid, width=w, height=h))
            print(f"  Generated {w}×{h} maze  |  🟢=start  🔴=end")
        elif cmd == "solve":
            if grid is None:
                print("  Generate a maze first."); continue
            path = solve_maze(grid, width, height)
            if path:
                print(render(grid, path, width, height))
                print(f"  Solved! Path length: {len(path)} steps")
            else:
                print("  No solution found.")
        elif cmd == "save":
            if grid is None:
                print("  Generate a maze first."); continue
            fname = input("  Filename [maze.txt]: ").strip() or "maze.txt"
            content = render(grid, path, width, height)
            Path(fname).write_text(content)
            print(f"  Saved to {fname}")
        else:
            print("  Commands: generate | solve | save | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Maze Generator")
    parser.add_argument("--width",  type=int, default=21, help="Width  (odd, default 21)")
    parser.add_argument("--height", type=int, default=11, help="Height (odd, default 11)")
    parser.add_argument("--solve",  action="store_true",  help="Show solution path")
    parser.add_argument("--save",   metavar="FILE",       help="Save maze to file")
    parser.add_argument("--seed",   type=int, default=None)
    args = parser.parse_args()

    if args.width or args.height:
        w = (args.width  | 1)
        h = (args.height | 1)
        grid = generate_maze(w, h, args.seed)
        path = solve_maze(grid, w, h) if args.solve else None
        output = render(grid, path, w, h)
        print(output)
        if path:
            print(f"Solution path: {len(path)} steps")
        if args.save:
            Path(args.save).write_text(output)
            print(f"Saved to {args.save}")
        return

    interactive()


if __name__ == "__main__":
    main()
