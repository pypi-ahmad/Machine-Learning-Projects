# Flashcard App GUI

Local Tkinter desktop app for creating, studying, and managing flashcard decks.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

## Data

The app stores deck data in `flashcards.json` beside `main.py`. It uses only the Python standard library and requires Python 3.13 or later.

Keep personal card content private. Deleting a deck or card permanently removes it from the local JSON file after confirmation.
