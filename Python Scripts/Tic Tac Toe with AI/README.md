# Tic Tac Toe with AI

> A feature-rich terminal Tic Tac Toe game with three play modes including an AI opponent.

## Overview

A command-line Tic Tac Toe game that supports Player vs. Player, Player vs. Computer (AI), and Computer vs. Computer modes. The AI uses a simple strategy that prioritizes winning moves, blocking opponent wins, corners, center, and edges in that order. The board layout mirrors a keyboard numpad (1–9).

## Features

- **Three game modes:**
  - Player vs. Computer (mode 0)
  - Player vs. Player (mode 1)
  - Computer vs. Computer (mode 2)
- AI opponent with strategic move selection (win/block/corner/center/edge priority)
- Random first-player selection
- Player name customization and X/O marker choice
- Side-by-side display of game board and available positions
- Win detection for rows, columns, and diagonals
- Draw detection when the board is full
- Replay option after each game
- Configurable via `.replit` for running on Replit

## Project Structure

```
Tic_tac_toe_with_ai/
├── tic-tac-toe-AI.py
├── tic_tac_toe.png
├── .replit
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `random` and `time` from the standard library)

## Installation

```bash
cd "Tic_tac_toe_with_ai"
```

No package installation required.

## Usage

```bash
python tic-tac-toe-AI.py
```

1. Select a game mode (0, 1, or 2).
2. Enter player names and choose X or O.
3. A random player is chosen to go first.
4. Enter positions 1–9 (numpad layout) when prompted.

### Board Layout

```
 7 | 8 | 9
-----------
 4 | 5 | 6
-----------
 1 | 2 | 3
```

## How It Works

1. **Board representation:** A list of 10 elements (index 0 unused), initialized to spaces.
2. **AI Strategy (`CompAI` function):**
   - Iterates through markers `O` then `X` in hardcoded order (not based on the AI's assigned marker): for each marker, checks if placing it in any open position would win, and returns the first winning position found.
   - Prefers open corners (1, 3, 7, 9), then center (5), then edges (2, 4, 6, 8).
3. **Display:** Shows the current board alongside available positions side-by-side.
4. **Win check:** Evaluates all 8 winning combinations (3 rows, 3 columns, 2 diagonals).
5. **Game loop:** Alternates turns between players until a win or draw, then offers replay.

## Configuration

- `.replit` file configured to run with `python tic-tac-toe-AI.py`.
- A `delay()` function is defined in the code for Computer vs. Computer mode but is never called, so there is no actual delay between moves.

## Limitations

- The AI is not a full minimax implementation — it uses heuristic priority ordering, so it is not guaranteed to play optimally in all situations.
- Non-numeric input at position prompt will crash (`int()` without try/except).
- In mode 0, the win message for the computer incorrectly says "THE Computer HAS WON" regardless of which player is the computer.

## License

Not specified.
