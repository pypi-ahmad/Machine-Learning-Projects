# Number Guessing Game

A local command-line game: guess the randomly selected whole number from 1 through 9.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Enter a whole number when prompted. The game tells you whether to guess higher or lower until you find the number.

## Behavior

- Invalid text and guesses outside the 1–9 range are rejected without counting as attempts.
- Every valid guess, including the winning guess, counts as an attempt.
- Each launch starts one game and does not save scores or use a network connection.
