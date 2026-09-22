# Rock Paper Scissors Game

An interactive command-line Rock Paper Scissors game. Choose the number of
decisive rounds, then play against a randomly selected computer choice.

## Run

```powershell
uv run python Rock_Paper_Scissors_Game.py
```

Enter a non-negative integer when prompted for the number of games. During a
round, enter a value beginning with `r`, `p`, or `s`; capitalization does not
matter. Invalid choices are prompted again without counting as a round.

```text
Enter the number of games you want to play: 3
User's Input: rock
Computer's Input: Scissors
```

Only decisive rounds count. A tie is shown and another round is requested, so
the number of choices entered may exceed the requested game count.

## Dependencies

The game uses only Python's standard library. uv records the required Python
version and supplies the project environment.

## Current limitations

- The game has no quit command or replay prompt once a series starts.
- An empty choice is not valid input.
- The script is designed to run directly, not to be imported as a library.
