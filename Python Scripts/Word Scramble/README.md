# Word Scramble

A local command-line word game. Unscramble randomly selected words, ask for hints, and track points across several rounds.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

The project uses only the Python standard library.

## Run

```powershell
cd "Python Scripts\Word Scramble"
uv run python .\main.py
```

Choose a difficulty and number of rounds:

```powershell
uv run python .\main.py --difficulty hard --rounds 10 --time 45
```

Set `--time 0` to play without a time limit:

```powershell
uv run python .\main.py --difficulty easy --rounds 3 --time 0
```

During a round, enter a guessed word, `hint` for a progressively revealing hint, or `skip` to reveal the answer.

## Notes

- Requested rounds are capped at the number of words available for the selected difficulty.
- Hints reduce the score for a correctly solved word.
- All game data is bundled locally; no network access is used.
