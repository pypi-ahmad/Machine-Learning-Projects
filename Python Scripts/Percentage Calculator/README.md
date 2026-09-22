# Percentage Calculator

An interactive terminal calculator for common percentage questions.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py
```

Choose one of the menu options to calculate:

- a percentage of a value;
- the percentage represented by one value of another;
- percentage change between two values;
- a percentage increase; or
- a percentage decrease.

Enter `0` to exit. The calculator rejects zero denominators for percentage and change calculations, and asks again after non-numeric input.

## Notes

This is a local, dependency-free CLI. It does not read files or make network requests.
