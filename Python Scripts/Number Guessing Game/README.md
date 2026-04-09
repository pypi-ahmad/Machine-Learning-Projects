# Number Guessing Game

> A command-line number guessing game where the player tries to guess a randomly generated number between 1 and 9.

## Overview

This is a classic number guessing game implemented in Python. The program generates a random integer between 1 and 9, then prompts the player to guess it. After each guess, the program provides directional hints (too high or too low) until the correct number is found, then displays the total number of attempts.

## Features

- Random number generation between 1 and 9
- Directional hints after each incorrect guess ("too high" or "too low")
- Tracks and displays the number of attempts upon winning
- Unlimited guesses until the correct answer

## Project Structure

```
Number_guessing_game/
├── main.py      # Game script
└── image.png    # Sample gameplay screenshot
```

## Requirements

- Python 3.x
- No external dependencies (uses only the `random` standard library module)

## Installation

No installation required.

```bash
cd Number_guessing_game
```

## Usage

```bash
python main.py
```

**Example session:**

```
Number guessing game
Guess a number (between 1 and 9):
5
Your guess was too high: Guess a number lower than 5
3
Your guess was too low: Guess a number higher than 3
4
CONGRATULATIONS! YOU HAVE GUESSED THE NUMBER 4 IN 2 ATTEMPTS!
```

## How It Works

1. Generates a random integer between 1 and 9 using `random.randint(1, 9)`
2. Initializes an attempt counter at 0
3. Enters a `while True` loop:
   - Reads user input and converts to `int`
   - If correct: prints congratulations with attempt count and breaks
   - If too low: prints hint to guess higher
   - If too high: prints hint to guess lower
   - Increments the attempt counter

## Limitations

- No input validation — entering a non-integer (e.g., letters) will crash with a `ValueError`
- The guessing range is very small (1–9), making the game trivial
- The attempt counter starts at 0 and increments only on wrong guesses, so the displayed count excludes the final correct guess
- No option to play again without restarting the script
- No difficulty levels or configurable range

## License

Not specified.
