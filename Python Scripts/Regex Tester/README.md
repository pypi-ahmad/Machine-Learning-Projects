# Regex Tester

An interactive terminal utility for testing Python regular expressions and substitutions.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py
```

Use the menu to test a pattern, perform a substitution, or view the built-in regex cheat sheet. End multiline input with a line containing `###`.

The tool uses Python's standard `re` module, runs locally, and makes no network requests.
