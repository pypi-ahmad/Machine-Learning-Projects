# Spaceship Game

> A two-player spaceship battle game built with Pygame where players shoot bullets to deplete each other's health.

## Overview

This is a local multiplayer spaceship battle game. Two players control yellow and red spaceships on opposite sides of a 900×500 window divided by a center border. Each player can move in four directions and fire bullets at the opponent. The first player to reduce the other's health to zero wins.

## Features

- **Two-player local multiplayer** with separate keyboard controls
- Health system (10 HP per player) with on-screen health display
- Bullet firing with a maximum of 3 bullets on-screen per player at a time
- Collision detection between bullets and spaceships
- Center border dividing the play area (players cannot cross to the other side)
- Winner announcement with a 5-second display before restarting
- Custom spaceship sprites and space background
- 60 FPS frame rate cap for consistent gameplay

## Project Structure

```
Spaceship_Game/
├── Assets/
│   ├── Red_Spaceship.png
│   ├── space.jpg
│   └── Yellow_Spaceship.png
├── main.py
├── requirements.txt
└── utility.py
```

## Requirements

- Python 3.x
- `pygame==2.0.1`

## Installation

```bash
cd "Spaceship_Game"
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

### Controls

**Yellow Player (Left Side):**

| Key | Action |
|-----|--------|
| W | Move up |
| S | Move down |
| A | Move left |
| D | Move right |
| Left Ctrl | Fire bullet |

**Red Player (Right Side):**

| Key | Action |
|-----|--------|
| ↑ (Up Arrow) | Move up |
| ↓ (Down Arrow) | Move down |
| ← (Left Arrow) | Move left |
| → (Right Arrow) | Move right |
| Right Ctrl | Fire bullet |

## How It Works

### Architecture

- **`main.py`**: Game loop, event handling, bullet logic, and rendering
- **`utility.py`**: Asset loading, movement handling, and winner display

### Game Flow

1. **Asset loading** (`utility.load_assests()`): Loads spaceship PNGs and space background, scales them to 50×40 pixels, and rotates them 90°/−90° to face each other
2. **Main loop** (`main()`):
   - Processes `QUIT` events and keyboard input for bullet firing
   - Custom Pygame events (`YELLOW_HIT`, `RED_HIT`) track bullet collisions
   - Health decrements on hit events
   - Checks for winner (health ≤ 0) and displays result via `utility.winner()`
3. **Movement** (`utility.yellow_handle_movement()` / `utility.red_handle_movement()`): Handles continuous key-press movement with boundary and border collision checks
4. **Bullets** (`handle_bullets()`): Moves bullets at velocity 7, checks rectangle collision with `colliderect()`, removes bullets that go off-screen or hit a target
5. **Rendering** (`drawWindow()`): Draws background, border, health text, spaceships, and bullets each frame

### Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| Window size | 900×500 | Game window dimensions |
| Spaceship size | 50×40 | Sprite dimensions |
| VELOCITY | 5 | Spaceship movement speed |
| BULLET_VEL | 7 | Bullet movement speed |
| MAX_BULLETS | 3 | Max bullets per player on screen |
| Health | 10 | Starting HP per player |
| FPS | 60 | Frame rate cap |

## Configuration

Edit values in `main.py` and `utility.py`:

- **Window size**: `width, height = 900, 500` in `utility.py`
- **Spaceship size**: `SPACESHIP_WIDTH, SPACESHIP_HEIGHT` in both files
- **Velocity/bullet speed**: `VELOCITY` and `BULLET_VEL` in `main.py`
- **Starting health**: `red_health` and `yellow_health` in `main()`
- **Assets**: Replace PNGs in `Assets/` folder

## Limitations

- After a winner is shown, `main()` is called recursively, which will cause a stack overflow after many rounds
- Movement uses `elif` chains, so only one direction can be processed per frame (no diagonal movement)
- No sound effects or music
- No single-player or AI mode
- Bullet count limit (`MAX_BULLETS = 3`) is defined inside the event loop rather than as a top-level constant
- The `load_assests()` function name contains a typo ("assests" instead of "assets")

## Security Notes

No security concerns.

## License

Not specified.
