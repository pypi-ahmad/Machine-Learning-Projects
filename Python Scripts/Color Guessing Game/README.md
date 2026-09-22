# Color Guessing Game

## Overview

Color Guessing Game is a terminal quiz for recognizing CSS color names, RGB values, and hexadecimal color codes.

## Setup and run

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run python main.py
uv run python main.py --mode rgb --rounds 10 --seed 42
uv run python main.py --mode hex --rounds 5
uv run python main.py --mode name --rounds 5
```

Available modes are `rgb`, `hex`, and `name`. Round counts must be at least one. `--seed` makes question ordering reproducible.

## Terminal note

The game uses ANSI true-color escape sequences for the color bar when the terminal supports them. The bar remains readable as ASCII characters in terminals without color support.

## Verification

Run `uv run python -m py_compile main.py` to check syntax.
