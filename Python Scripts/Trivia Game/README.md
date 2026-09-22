# Trivia Game

A terminal multiple-choice trivia game with science, history, and technology categories, three difficulty levels, optional timers, and score tracking.

## Run it

```powershell
uv sync
uv run python main.py
```

You can also start a specific round:

```powershell
uv run python main.py --category science --difficulty hard --questions 3
uv run python main.py --category technology --difficulty easy --timed
```

Enter an answer from `1` to `4`. If requested questions exceed the category pool, the game uses every available question once.

## Dependencies

- Python 3.14+
- No third-party packages
