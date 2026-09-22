# Decimal Binary Converter

A dependency-free CLI for converting one signed integer between decimal and binary notation.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run python decimal_to_binary.py
```

Choose a conversion direction, then enter an integer. Binary inputs accept only `0` and `1`, with an optional leading `+` or `-`.

## Examples

```text
Decimal: -5
Binary: -101

Binary: 101
Decimal: 5
```

## Project files

```text
decimal_to_binary.py  # CLI entry point and conversion functions
output.png            # Existing project screenshot
pyproject.toml        # uv project definition
uv.lock               # Resolved Python environment
```

## Verification

```powershell
uv run python -m py_compile decimal_to_binary.py
uv lock --check
```
