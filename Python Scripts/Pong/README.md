# Pong

A local two-player Pong game built with Tkinter.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py
```

Player 1 uses `W` and `S`. Player 2 uses the up and down arrow keys. Press `Space` to serve or pause, and `R` to restart after a match. The first player to reach 10 points wins.

Play against the computer or adjust the starting ball speed:

```powershell
uv run --no-config python main.py --vs-ai --speed 6
```

## Notes

The game uses only the Python standard library, runs locally, and does not create files or make network requests.
