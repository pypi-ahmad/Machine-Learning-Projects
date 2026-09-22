# Flashcard App

Local Streamlit app for creating and studying small flashcard decks.

## Run

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

## Data

The app stores each deck as JSON in `flashcard_decks/` beside `main.py`. The first launch creates a sample Python deck. Keep personal card content private and review it before sharing the folder.

## Notes

- Python 3.13 or later is required.
- Deck names cannot include file paths.
- The app does not sync cards to an external service.
