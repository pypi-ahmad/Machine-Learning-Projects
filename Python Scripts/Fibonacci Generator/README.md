# Fibonacci Generator

A dependency-free interactive explorer for Fibonacci, Lucas, Tribonacci, and golden-ratio sequences.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Choose an option from the menu to:

- Generate the first `N` Fibonacci numbers.
- Find a specific Fibonacci term using matrix exponentiation.
- Check whether a non-negative integer is in the Fibonacci sequence.
- Inspect Fibonacci properties, Lucas numbers, Tribonacci numbers, or golden-ratio convergents.

## Limits

The interactive nth-term command caps `N` at 20,000 so formatted integer output remains practical in the standard Windows console. Fibonacci values grow quickly, so use smaller sequence lengths when you want to inspect every value.
