# Rock Paper Scissors

A command-line Rock Paper Scissors game with score tracking, win and loss
streaks, and an optional best-of series.

## Run

```powershell
uv run python main.py
```

Use `r`, `p`, or `s` to play a round. Enter `q` to finish and view the final
score.

## Best-of series

```powershell
uv run python main.py --rounds 5
```

The first player to win a majority of the requested rounds wins the series.

## Dependencies

This project uses only the Python standard library. uv manages the required
Python version and provides the reproducible project environment.
