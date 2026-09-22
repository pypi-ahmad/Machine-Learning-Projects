# Password Generator

Generate local passwords or word-based passphrases with Python's `secrets` module.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py 20
```

Use options to omit character sets or generate multiple values:

```powershell
uv run --no-config python main.py 16 --no-symbols --count 3
uv run --no-config python main.py --passphrase --words 4
```

Run without a length to use the interactive menu.

## Behavior and safety

- Passwords are generated locally with cryptographically secure randomness and are only printed to the terminal.
- When character sets are enabled, each selected set contributes at least one character. Length must be large enough for those sets.
- The `exclude similar` option removes `il1Lo0O` before required characters are chosen.
- The built-in passphrase word list is intentionally small and is suitable for demonstrations, not as a replacement for a larger Diceware-style list.
- Do not paste generated passwords into logs, shell history, source files, or shared documents.
