# Prime Number Tools

An interactive terminal toolkit for common prime-number calculations.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py
```

Use the menu to check primality, generate primes up to a limit, factor a number, calculate GCD and LCM, list twin-prime pairs, find an nth prime, or show Goldbach pairs for an even number. Enter `0` to exit.

## Notes

The tool limits interactive prime generation to 10,000,000, twin-prime searches to 1,000,000, and nth-prime searches to 100,000 to keep calculations practical.

It runs locally with only the Python standard library and makes no network requests.
