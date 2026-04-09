# Snake Game (GUI)

> A classic Snake game built with Pygame featuring score tracking and replay functionality.

## Overview

This is a graphical Snake game implemented using Pygame. The player controls a snake that moves around a 600×400 window, eating food to grow longer. The game ends when the snake hits a wall or its own body, with an option to replay.

## Features

- Smooth snake movement with arrow key controls
- Food spawning at random grid-aligned positions
- Score display in the top-left corner
- Wall collision detection (game over when hitting boundaries)
- Self-collision detection (game over when snake crosses itself)
- Game over screen with "Play Again" (`C`) or "Quit" (`Q`) options
- Configurable snake speed (15 FPS by default)

## Project Structure

```
Snake Game (GUI)/
└── game.py
```

## Requirements

- Python 3.x
- `pygame`

## Installation

```bash
cd "Snake Game (GUI)"
pip install pygame
```

## Usage

```bash
python game.py
```

### Controls

| Key | Action |
|-----|--------|
| ↑ (Up Arrow) | Move up |
| ↓ (Down Arrow) | Move down |
| ← (Left Arrow) | Move left |
| → (Right Arrow) | Move right |
| C | Play again (on game over screen) |
| Q | Quit (on game over screen) |

## How It Works

1. **Initialization**: Creates a 600×400 pixel Pygame window with a blue background
2. **Game loop** (`gameLoop()`):
   - Processes keyboard events for directional movement
   - Updates snake head position by `snake_block` (10px) per frame
   - Checks for wall collisions (boundaries of the display)
   - Checks for self-collision (head overlapping any body segment)
   - When the snake head position matches the food position, the snake length increases and food respawns randomly
3. **Rendering**: Each frame draws the blue background, green food rectangle, black snake body rectangles, and the yellow score text
4. **Game over**: Displays a red message on a blue screen; pressing `C` restarts by recursively calling `gameLoop()`

### Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `dis_width` | 600 | Window width in pixels |
| `dis_height` | 400 | Window height in pixels |
| `snake_block` | 10 | Size of each snake segment in pixels |
| `snake_speed` | 15 | Game speed (frames per second) |

## Configuration

Edit these variables in `game.py` to customize:

- **`dis_width` / `dis_height`**: Window dimensions
- **`snake_block`**: Snake and food size
- **`snake_speed`**: Game speed (FPS)
- **Color tuples**: `white`, `yellow`, `black`, `red`, `green`, `blue`

## Limitations

- Replay is implemented by recursively calling `gameLoop()`, which will cause a stack overflow after many replays
- No pause functionality
- No high score persistence
- No difficulty scaling (speed remains constant)
- Snake cannot reverse direction (no explicit prevention, but grid-aligned movement mitigates it)
- Food can spawn on the snake's body

## Security Notes

No security concerns.

## License

Not specified.
