# Brick Breaker Game

A classic Breakout-style brick breaker game built with Pygame. The player controls a paddle to deflect a ball and destroy a wall of colored bricks.

## Overview

This is a **GUI game** built with Pygame. The player moves a paddle horizontally using arrow keys to keep a bouncing ball in play. The ball destroys bricks on contact, with bricks having different durability levels based on their row. The game ends when all bricks are destroyed (win) or the ball falls below the paddle (lose).

## Features

- 6x6 grid of colored bricks with variable durability (1–3 hit points based on row)
- Ball physics with speed management and wall/paddle/brick collision detection
- Paddle movement via left/right arrow keys with speed influence on ball angle
- Three brick strength tiers: orange (3 hits), white (2 hits), green (1 hit)
- Win/lose detection with restart capability (click anywhere to restart)
- 500x500 pixel game window at 60 FPS
- Visual feedback with outlined bricks and ball

## Dependencies

> *Inferred from imports (no `requirements.txt` present)*

- `pygame`

## How It Works

1. **Initialization:** A 500x500 Pygame window is created. A `Block` object generates a 6x6 grid of bricks, each with a hit-point value (rows 0–1: 3 HP, rows 2–3: 2 HP, rows 4–5: 1 HP).
2. **Ball class:** Manages ball position, speed, and collision logic. The ball bounces off walls, the paddle, and bricks. When a brick is hit, its HP decreases; at 0 HP, it is removed.
3. **base class:** The paddle (width = window_width / 6) moves left/right via keyboard input. Its movement direction influences ball angle on collision.
4. **Game loop:** Runs at 60 FPS. Updates paddle position, ball motion, and collision detection each frame. Displays win/lose messages when appropriate.
5. **Restart:** Clicking anywhere after a win or loss resets the ball, paddle, and bricks.

## Project Structure

```
Brick Breaker game/
├── brick_breaker.py   # Complete game implementation
└── Readme.md          # This file
```

## Setup & Installation

```bash
pip install pygame
```

## How to Run

```bash
cd "Brick Breaker game"
python brick_breaker.py
```

## Configuration

No external configuration. Game parameters are defined as constants at the top of the script:

- `Window_width` / `Window_height` — 500x500
- `game_rows` / `game_coloumns` — 6x6 brick grid
- `frame_rate` — 60 FPS
- Ball speed — initial 4, max 5

## Testing

No formal test suite present.

## Limitations

- Window size and grid dimensions are hardcoded (not configurable without editing the script).
- No score tracking or display.
- No levels or increasing difficulty.
- The `Block` class instance shadows the `Block` class name (line: `Block = Block()`), preventing re-instantiation.
- No sound effects.
- The paddle collision logic has an `else` branch that inverts `x_speed` instead of `y_speed`, which may cause unexpected ball behavior on edge hits.
