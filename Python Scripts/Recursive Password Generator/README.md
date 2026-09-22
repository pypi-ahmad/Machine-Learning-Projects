# Recursive Password Generator

Generate one secure random password from letters, digits, and punctuation.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python generator.py 16
```

Passwords use Python's `secrets` module and exclude whitespace/control characters. They are printed only and are not saved, copied, or transmitted.
