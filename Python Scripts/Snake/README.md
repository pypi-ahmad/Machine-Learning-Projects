# Snake

A Tkinter implementation of the classic Snake game. Eat the food to grow,
avoid the walls and the snake's body, and restart after a game over.

## Run

```powershell
uv run python main.py
```

## Controls

- Arrow keys: change direction
- `P`: pause or resume
- `R`: restart

The game awards 10 points for each food item and keeps the high score for the
current application session. It requires an interactive desktop session with
Tkinter support.

## Dependencies

The project uses only Python's standard library, including Tkinter. uv records
the Python requirement and provides the reproducible environment.
