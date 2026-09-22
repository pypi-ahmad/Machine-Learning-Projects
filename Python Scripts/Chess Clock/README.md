# Chess Clock

## Overview

Chess Clock is a terminal clock for two players. It supports named presets, custom starting time, and Fischer-style per-move increments.

## Setup and run

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run python main.py --preset blitz
uv run python main.py --minutes 5 --increment 3 --white Alice --black Bob
```

With no arguments, the program prompts for a preset or custom time control. During a game, press Enter after each move, `P` to pause or resume, and `Q` to stop the game.

## Time controls

- `bullet`: 1+0
- `blitz`: 5+0
- `rapid`: 10+0
- `classical`: 60+0
- `fischer`: 5+3
- `bronstein`: 10+2

## Verification

Run `uv run python -m py_compile main.py` to check syntax.
