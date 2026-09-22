# Snake Game GUI

A classic Pygame Snake game. Guide the snake to food, avoid the walls and its
own body, then restart or quit from the game-over screen.

## Run

```powershell
uv sync
uv run python game.py
```

## Controls

- Arrow keys: move the snake
- `C`: restart after game over
- `Q`: quit after game over

The game window is 600 by 400 pixels. Food adds one snake segment and one
point; the default speed is 15 frames per second.

## Implementation notes

Pygame setup occurs only inside `main()`, so importing `game.py` does not open
a window. Restarting returns to the outer game loop instead of recursively
calling the game loop, avoiding growth of the call stack across replays.

## Dependencies

This project pins Python to the 3.13 series for the installed Windows Pygame
wheel. uv manages Pygame in `pyproject.toml` and locks the resolved version in
`uv.lock`.
