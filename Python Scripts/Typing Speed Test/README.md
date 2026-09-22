# Typing Speed Test

A terminal typing exercise that measures raw and net WPM, character-level accuracy, errors, and elapsed time across easy, medium, hard, or custom text.

## Run it

```powershell
uv sync
uv run python main.py
```

Choose a mode, press Enter when you are ready, then type the displayed text and press Enter again to finish. The timer starts immediately after the ready prompt.

## History

Completed tests are stored in `typing_history.json` beside `main.py`; the latest 50 results are retained. The file stays local and is created only after a completed test. Back it up if you want to keep the history, and do not commit it if it contains information you want to keep private.

## Dependencies

- Python 3.14+
- No third-party packages
