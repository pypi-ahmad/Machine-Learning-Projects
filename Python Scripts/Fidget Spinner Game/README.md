# Fidget Spinner Game

A small desktop fidget-spinner simulation built with Python's standard-library `turtle` graphics module.

## Requirements

- Python 3.13 or later with Tkinter/turtle support
- [uv](https://docs.astral.sh/uv/)

## Run

```powershell
uv sync --no-config
uv run --no-config python "Fidget Spinner Game in Python.py"
```

Press the spacebar to add momentum. The spinner slows gradually between flicks, and momentum is capped so repeated key presses remain responsive.

Close the turtle window normally to quit. The game has no external dependencies, persistence, or network access.
