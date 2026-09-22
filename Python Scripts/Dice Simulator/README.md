# Dice Simulator

A local Tkinter desktop app for rolling one to six standard six-sided dice. It animates each roll and keeps a session-only history with basic statistics.

## Requirements

- Python 3.13+ with Tkinter available
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run python main.py
```

Choose the number of dice and an optional modifier, then select **Roll**. The history and statistics exist only while the app remains open; no rolls are written to disk.

## Behavior

- Rolls one to six d6 dice and adds a modifier from -20 to 20.
- Animates the displayed d6 faces.
- Retains up to 50 history entries for the active session.
- Displays count, average, minimum, maximum, and a simple face-frequency view.
- Safely treats an incomplete or out-of-range editable dice-count value as a valid in-range value.

## Project files

```text
main.py         # Tkinter application and dice logic
pyproject.toml  # uv project definition
uv.lock         # Resolved Python environment
```

## Verification

```powershell
uv run python -m py_compile main.py
uv lock --check
```
