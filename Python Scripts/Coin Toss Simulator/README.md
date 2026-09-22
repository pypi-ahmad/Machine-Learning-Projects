# Coin Toss Simulator

## Overview

Coin Toss Simulator is a terminal tool for fair or weighted coin-toss experiments. It reports heads, tails, runs, longest streaks, and optional streak-frequency analysis.

## Setup and run

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run python main.py --flips 100
uv run python main.py --flips 1000 --bias 0.6 --seed 42
uv run python main.py --flips 50 --streak --seed 42
```

Run without options for interactive mode. `--bias` is the probability of heads and must be between 0 and 1, inclusive.

## Verification

Run `uv run python -m py_compile main.py` to check syntax. The `--seed` option makes command-line simulations reproducible.
