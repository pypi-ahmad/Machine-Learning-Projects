# Hangman Game

> A graphical Hangman word-guessing game built with Pygame.

## Overview

A classic Hangman game with a graphical interface where players guess letters by clicking on-screen buttons. The game displays a hangman figure that progressively builds with each wrong guess, and the word is revealed letter by letter as correct guesses are made.

## Features

- Graphical interface using Pygame with clickable letter buttons
- Visual hangman progression through 7 image stages (0–6 wrong guesses)
- Random word selection from a word list file
- Win/lose detection with on-screen messages
- Color-themed UI (blue background, pink end screen)
- Letters disappear after being clicked to prevent duplicate guesses

## Project Structure

```
Hangman-Game/
├── main.py          # Main game logic and Pygame rendering
├── words.txt        # Comma-separated word list
├── hangman0.png     # Hangman stage images (0 = empty gallows)
├── hangman1.png
├── hangman2.png
├── hangman3.png
├── hangman4.png
├── hangman5.png
├── hangman6.png     # Full hangman (game over)
└── README.md
```

## Requirements

- Python 3.x
- `pygame`

## Installation

```bash
cd "Hangman-Game"
pip install pygame
```

## Usage

```bash
python main.py
```

Click on letters to guess. The game ends when you either guess the full word or accumulate 6 wrong guesses.

## How It Works

1. **Initialization**: Sets up an 800×500 Pygame window with 26 clickable circular letter buttons arranged in two rows.
2. **Word Selection**: Reads `words.txt` (comma-separated), splits into a list, and picks a random word converted to uppercase.
3. **Game Loop**: Runs at 60 FPS. On each mouse click, calculates distance to each visible letter button. If a click lands within the button radius, the letter is marked as guessed.
4. **Drawing**: Each frame renders the title, the word with blanks for unguessed letters, visible letter buttons, and the current hangman image.
5. **Win/Lose Check**: After each guess, checks if all letters in the word have been guessed (win) or if `hangman_status` has reached 6 (lose). Displays a message for 3 seconds, then exits.

## Configuration

- **Image paths**: Hardcoded as `./Hangman-Game/hangman{0-6}.png` — must be run from the parent directory or paths must be updated.
- **Word list path**: Hardcoded as `./Hangman-Game/words.txt`.
- **Window size**: 800×500 pixels, set via `WIDTH` and `HEIGHT` constants.
- **Fonts**: Uses `comicsans` system font.

## Limitations

- Hardcoded relative paths require the script to be run from a specific working directory.
- No restart mechanism — the game exits after a win or loss and must be re-run.
- No difficulty levels or word categories.
- The variable `i` is reused for both the event loop and letter iteration, which could cause subtle bugs.
- Word list contains a trailing space on "zodiac " which could cause matching issues.

## Security Notes

No security concerns identified.

## License

Not specified.
