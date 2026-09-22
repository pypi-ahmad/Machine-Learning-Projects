# Emoji Guessing

A small command-line game: identify movies, food, animals, or phrases from emoji clues.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Choose a category when prompted. During a round, enter your answer, `hint` for an initial-letter hint, or `skip` to reveal the answer.

## Options

Start a category directly:

```powershell
uv run --no-config python main.py --category movies --rounds 5
```

Disable hints or replay a known puzzle order:

```powershell
uv run --no-config python main.py --category animals --rounds 3 --no-hints --seed 42
```

Available categories are `movies`, `food`, `animals`, `phrases`, and `all`.
