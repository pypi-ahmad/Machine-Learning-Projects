# Trigonometry Helper

A terminal calculator for trigonometric functions, inverse values, common unit-circle values, angle conversion, and triangles specified by three sides.

## Run it

```powershell
uv sync
uv run python main.py
```

Examples:

```powershell
uv run python main.py --angle 45 --unit deg
uv run python main.py --inv 0.5
uv run python main.py --circle
uv run python main.py --triangle --sides 3 4 5
```

The interactive prompt provides the same calculations. Values printed as `undefined` have no finite real result at that angle.

## Dependencies

- Python 3.14+
- No third-party packages
