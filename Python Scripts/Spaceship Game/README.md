# Spaceship Game

Local two-player Pygame battle game. Each player moves within one half of the screen and fires at the opponent.

## Setup

Requirements: Python 3.13+ and a desktop session that can open a Pygame window.

```powershell
cd "Spaceship Game"
uv sync
```

## Run

```powershell
uv run python main.py
```

Controls:

| Player | Move | Fire |
| --- | --- | --- |
| Yellow, left side | W, A, S, D | Left Ctrl |
| Red, right side | Arrow keys | Right Ctrl |

Each player starts with 10 health and can have at most three bullets in flight. The winner screen appears for five seconds, then a new round begins. Close the game window to exit.

## Verification mode

Load assets and exit without starting a game round:

```powershell
uv run python main.py --smoke
```

## Behavior

The game uses a 60 FPS loop, supports diagonal movement, loads assets relative to `main.py`, and restarts rounds iteratively rather than recursively. It does not access the network or write files.

## Project files

```text
Spaceship Game/
├── Assets/
├── main.py
├── pyproject.toml
└── uv.lock
```
