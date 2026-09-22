# Random Password Generator

Two local password generators that use Python's cryptographically secure `secrets` module.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python python-password-generator.py
uv run --no-config python random_password_gen.py 16
```

The first command creates a 16-character password. The second accepts a length of at least 3 and mixes letters, numbers, and `@#$%&*` symbols.

Passwords are printed only; the project does not save, copy, transmit, or validate them.
