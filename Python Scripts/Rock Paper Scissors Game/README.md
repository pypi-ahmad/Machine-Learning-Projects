# Rock Paper Scissors Game

> A command-line Rock Paper Scissors game where the user plays against the computer over a configurable number of rounds.

## Overview

An interactive CLI game implemented in Python. The player specifies how many rounds to play, then inputs R (Rock), P (Paper), or S (Scissors) each round. The computer picks randomly. Scores are tracked and displayed after each round and at the end of the game.

## Features

- Configurable number of game rounds
- Flexible input parsing — accepts any string starting with R, P, or S (case-insensitive)
- Tracks and displays running score after each round
- Final result announcement (win, lose, or tie)
- Input validation with re-prompt on invalid input
- Tie rounds do not count toward the total game count

## Project Structure

```
RockPaperScissors_Game/
├── Rock_Paper_Scissors_Game.py   # Main game script
└── README.md
```

## Requirements

- Python 3.x
- `random` (Python standard library)

## Installation

```bash
cd "RockPaperScissors_Game"
```

No external dependencies required.

## Usage

```bash
python Rock_Paper_Scissors_Game.py
```

**Example interaction:**

```
Enter the number of games you want to play: 3

User's Input: rock
Computer's Input:  Scissors

SCORE:
User Score: 1	Computer Score: 0
```

The game accepts inputs like `rock`, `R`, `Paper`, `p`, `scissors`, etc. — only the first character matters (converted to uppercase).

## How It Works

1. The user enters the total number of rounds to play.
2. Each round: the user inputs their choice, which is trimmed to the first character and uppercased.
3. The input is validated against the dictionary keys (`R`, `P`, `S`). Invalid inputs trigger a re-prompt without consuming a round.
4. The computer's choice is selected randomly from the same key set using `random.choice()`.
5. Win/loss is determined by standard Rock-Paper-Scissors rules. Ties are announced but don't increment either score.
6. The loop continues until `user_count + comp_count` equals the requested number of games (ties extend play).
7. Final scores and the overall winner are displayed.

## Configuration

No configuration needed. Everything is handled via interactive prompts.

## Limitations

- Tie rounds do not count toward the game count, so the actual number of inputs may exceed the requested number of games.
- No option to quit mid-game.
- No replay option — must re-run the script to play again.
- The score display uses tab-based formatting that may misalign in some terminals.

## Security Notes

No security concerns.

## License

Not specified.


