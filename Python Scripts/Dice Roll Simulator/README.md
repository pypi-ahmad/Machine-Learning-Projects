# Dice Roll Simulator

A dependency-free CLI dice roller that supports standard `NdS` notation, modifiers, advantage or disadvantage, dropped low rolls, and simple distribution statistics.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run python main.py "2d6+3"
```

Start interactive mode by omitting the notation:

```powershell
uv run python main.py
```

## Examples

```powershell
# Deterministic result for a demonstration or repeatable check
uv run python main.py "1d20+5" --adv --seed 42

# Roll four dice and drop the lowest result
uv run python main.py "4d6" --drop 1

# View a simulated distribution for one six-sided die
uv run python main.py --stats 6 --seed 42
```

Use only one of `--adv` and `--dis`. `--drop` must be at least zero and smaller than the number of dice.

## Project files

```text
main.py         # CLI entry point
pyproject.toml  # uv project definition
uv.lock         # Resolved Python environment
```

## Verification

```powershell
uv run python main.py "2d6+3" --seed 42
uv run python -m py_compile main.py
uv lock --check
```
