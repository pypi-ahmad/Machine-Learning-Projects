# Random Word from List

Select one non-empty line from a UTF-8 text file.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python Random_word_from_list.py
uv run --no-config python Random_word_from_list.py .\my-words.txt
```

Without an argument, the script reads the bundled `file.txt` word list. It runs locally, does not modify the source list, and makes no network requests.
