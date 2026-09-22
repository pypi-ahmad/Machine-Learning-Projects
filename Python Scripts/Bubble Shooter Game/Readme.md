# Bubble Shooter Game

The project uses `pygame-ce`, the Python 3.14-compatible Pygame distribution,
while retaining the standard `pygame` import API.

A bubble shooter game built with Pygame, with color-matching bubble popping, score tracking, background music, and a rotatable arrow launcher.

## Overview

This is a **GUI game** built with Pygame. The player aims and shoots bubbles with a rotatable arrow. Three or more connected bubbles of the same color pop and award points. Floating bubbles are also removed. The player wins by clearing all bubbles and loses if bubbles reach the bottom of the screen.

## Features

- Rotatable arrow aiming system controlled by left/right arrow keys
- Bubble launching with spacebar
- Color-matching logic: 3+ connected same-color bubbles pop on contact
- Floating bubble detection and removal (bubbles not connected to the top row are cleared)
- Score system: 10 points per bubble popped
- Next-bubble preview display
- Background music with multiple tracks (auto-advances through playlist)
- Pop sound effect on bubble destruction
- Win/lose conditions with end screen and replay option
- 940x740 pixel game window at 120 FPS
- 14 distinct bubble colors
- 25-column by 20-row bubble grid with offset rows (honeycomb layout)

## Dependencies

> *Inferred from imports (no `requirements.txt` present)*

- `pygame`

## How it works

1. **Initialization:** A 940x740 Pygame window is created. A 25x20 grid (`bbarr`) is initialized; the top 5 rows are filled with randomly colored bubbles in a honeycomb pattern (odd rows offset by one bubble radius).
2. **Arrow (`Ary` class):** Loads `Arrow.png`, rotates based on left/right key input, and displays the aiming direction.
3. **Bubble (`Bubble` class):** Represents each bubble sprite. When launched, it moves based on its angle using trigonometric calculations. Bounces off left/right walls.
4. **Collision & Placement (`stbb`):** When a launched bubble collides with an existing bubble or reaches the top, it snaps to the nearest grid position. The `popbb()` function recursively finds connected same-color bubbles. If 3+ are found, they are removed and a pop sound plays.
5. **Floating Bubble Removal (`chkfflotrs`):** After popping, `chkfflotrs()` checks which bubbles are still connected to the top row and removes any floating disconnected bubbles.
6. **Score (`Score` class):** Tracks and displays the score, adding 10 points per popped bubble.
7. **Music:** Cycles through 3 OGG music tracks; plays `popcork.ogg` as a pop sound effect.
8. **End Screen:** Displays win/lose message with the final score. Press Enter to replay or Escape to quit.

## Project Structure

```
Bubble Shooter Game/
├── bubbleshooter.py              # Complete game implementation (~596 lines)
├── Arrow.png                     # Arrow/aiming cursor sprite
├── bgmusic.ogg                   # Background music track
├── Goofy_Theme.ogg               # Background music track
├── Whatever_It _Takes_OGG.ogg    # Background music track
├── popcork.ogg                   # Bubble pop sound effect
├── bubbleshoot.gif               # Game screenshot/preview
└── Readme.md                     # This file
```

## Setup & Installation

```bash
uv sync
```

Ensure all asset files (`Arrow.png`, `*.ogg`) are in the same directory as the script.

## How to Run

```bash
cd "Bubble Shooter Game"
uv run python bubbleshooter.py
```

### Controls

| Key | Action |
|-----|--------|
| Left Arrow | Rotate arrow left |
| Right Arrow | Rotate arrow right |
| Spacebar | Shoot bubble |
| Escape | Quit game |
| Enter | Restart (on end screen) |

## Configuration

No external configuration files. Key constants are defined at the top of `bubbleshooter.py`:

- `FPS` — 120 frames per second
- `winwdth` / `winhgt` — 940x740 window size
- `bubblerad` — 20 pixel bubble radius
- `bubblelyrs` — 5 initial layers of bubbles
- `arywdth` / `aryhgt` — 25x20 grid dimensions
- `clrlist` — 14-color palette for bubbles
- `musclist` — 3 background music tracks

## Testing

No formal test suite present.

## Limitations

- All game constants (window size, grid size, FPS, colors) are hardcoded.
- Variable names are heavily abbreviated (e.g., `bbarr`, `stbb`, `popbb`, `chkfflotrs`), reducing readability.
- No difficulty progression or levels.
- No high-score persistence between sessions.
- The recursive `popbb()` and `popflotrs()` functions could hit Python's recursion limit on very large grids.
- Asset files (`Arrow.png`, `*.ogg`) must be in the working directory — no path resolution is performed.
- No pause functionality.
