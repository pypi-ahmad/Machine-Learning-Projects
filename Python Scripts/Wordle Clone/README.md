# Wordle Clone

A local terminal Wordle-style game. Guess a five-letter word in six tries and use the colored tiles and keyboard to narrow down the answer.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

The project uses only the Python standard library.

## Run

```powershell
cd "Python Scripts\Wordle Clone"
uv run python .\main.py
```

Choose normal mode or hard mode from the menu. In hard mode, green letters from the preceding guess must remain in the same positions.

## Feedback

- Green: correct letter in the correct position.
- Yellow: correct letter in a different position.
- Grey: letter is not available in the remaining target letters.

The game uses a bundled word list and does not make network requests.
