# Reaction Timer

An interactive terminal game for measuring reaction time across multiple rounds.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py --rounds 5
```

Wait for the signal, then press Enter as quickly as possible. The summary reports average, best, worst, standard deviation, and a rough benchmark label.

## Notes

Terminal input and operating-system scheduling affect the result, so this is a casual practice tool rather than a scientific measurement. It runs locally with the Python standard library and makes no network requests.
