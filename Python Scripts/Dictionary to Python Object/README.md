# Dictionary to Python Object

A small dependency-free helper that wraps nested dictionaries so their values can be read with attribute syntax.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run the example

From this directory:

```powershell
uv sync
uv run python conversion.py
```

The included example prints:

```text
a=5, c.d=8
```

## Use as a helper

```python
from conversion import obj

data = {"a": 5, "b": 7, "c": {"d": 8}}
converted = obj(data)
print(converted.a)
print(converted.c.d)
```

Keys must be non-private, valid Python identifiers. The helper raises `ValueError` for keys that cannot be accessed safely as attributes. It converts nested dictionaries, but intentionally leaves lists and other values unchanged.

## Project files

```text
conversion.py   # Object wrapper and example entry point
pyproject.toml  # uv project definition
uv.lock         # Resolved Python environment
```

## Verification

```powershell
uv run python conversion.py
uv run python -m py_compile conversion.py
uv lock --check
```
