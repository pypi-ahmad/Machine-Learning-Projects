# Terminal-Based Hangman Game

> A classic hangman word-guessing game played entirely in the terminal, with ASCII art gallows and a large word list.

## Overview

This is a terminal-based implementation of the classic Hangman game. A random word is selected from a JSON file containing over 1,400 words. The player guesses one letter at a time (or attempts the full word) with 6 allowed wrong guesses. ASCII art displays the hangman's progressive state after each guess.

## Features

- Random word selection from a JSON word list (~1,400+ words)
- Single letter guessing with duplicate detection
- Full word guessing support
- 6 wrong guesses allowed before game over
- ASCII art hangman display (7 stages from empty gallows to full figure)
- Displays word length hint throughout the game
- Tracks guessed letters and guessed words separately
- Play-again prompt after each game
- Input validation (alphabetic characters only)

## Project Structure

```
Terminal_Based_Hangman_Game/
├── hangman.py
├── words.json
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `random` and `json` from the standard library)

## Installation

```bash
cd Terminal_Based_Hangman_Game
```

No additional installation needed — standard library only.

## Usage

```bash
python hangman.py
```

**Gameplay:**

1. The game selects a random word and displays blanks (`_`) for each letter
2. The word length is shown as a hint
3. Enter a single letter to guess, or type the entire word
4. Correct letters are revealed in place; wrong guesses reduce remaining tries
5. ASCII art updates after each guess to show the hangman's state
6. Win by revealing all letters or guessing the full word; lose after 6 wrong guesses
7. After each game, you're prompted to play again (Y/N)

**Example:**

```
Let's play Hangman!
     --------
     |      |
     |
     |
     |
     |
     -
_ _ _ _ _ _ _

Length of the word:  7

Please guess a letter or the word: e
Good job, e is in the word!
```

## How It Works

1. `get_word()` loads `words.json`, picks a random word from the `"word_list"` array, and converts it to uppercase
2. `play(word)` runs the main game loop:
   - Initializes blank completion string, tried letters/words lists, and tries counter (6)
   - On each turn, accepts a letter or full-word guess
   - Single-letter guesses: checks for duplicates, validates against the word, reveals matching positions
   - Full-word guesses: must match word length, checked for duplicates, compared to the answer
   - Calls `display_hangman(tries)` to render the current ASCII art stage
3. `display_hangman(tries)` returns one of 7 pre-defined ASCII art strings (index 0 = full body/game over, index 6 = empty gallows)
4. `main()` loops the game with a play-again prompt

## Configuration

- Word list: `words.json` — a JSON file with a `"word_list"` array of strings (~1,400+ English words)
- Tries: Hardcoded to 6 wrong guesses in `hangman.py`

## Limitations

- The word list is English-only and contains only common words
- No difficulty levels or category selection
- No score tracking across games
- All words are converted to uppercase — no mixed-case support
- The `words.json` file must be in the same directory as `hangman.py` (relative path)
- No clear-screen between turns, so the terminal scrolls with repeated output

## Security Notes

No security concerns identified.

## License

Not specified.
