# Password Strength Checker

An interactive local tool that scores passwords against length, character-set, common-password, repetition, and simple-sequence checks.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Passwords are requested with a hidden terminal prompt when supported. The batch option accepts one password per hidden prompt and stops on a blank entry.

## Behavior and limits

- The checker runs locally and does not save or send entered passwords.
- Entropy is a simple character-set estimate, not a measurement of real-world resistance to guessing attacks.
- The common-password and sequence lists are intentionally small. Passing the checks does not guarantee that a password is safe.
- Use a unique, randomly generated password for every account and enable multi-factor authentication where available.
