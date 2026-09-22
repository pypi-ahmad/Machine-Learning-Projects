"""A small command-line number guessing game."""

import random


LOWER_BOUND = 1
UPPER_BOUND = 9


def play_game(number: int) -> None:
    """Run one game for a preselected target number."""
    attempts = 0
    print("Number guessing game")
    print(f"Guess a number between {LOWER_BOUND} and {UPPER_BOUND}.")

    while True:
        answer = input("Guess: ").strip()
        try:
            guess = int(answer)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if not LOWER_BOUND <= guess <= UPPER_BOUND:
            print(f"Choose a number from {LOWER_BOUND} to {UPPER_BOUND}.")
            continue

        attempts += 1
        if guess == number:
            print(f"Congratulations! You guessed {number} in {attempts} attempt(s).")
            return
        if guess < number:
            print(f"Too low. Guess higher than {guess}.")
        else:
            print(f"Too high. Guess lower than {guess}.")


def main() -> None:
    play_game(random.randint(LOWER_BOUND, UPPER_BOUND))


if __name__ == "__main__":
    main()
