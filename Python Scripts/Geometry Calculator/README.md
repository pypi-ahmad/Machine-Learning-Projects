# Geometry Calculator

Interactive command-line calculator for common 2D and 3D shapes plus coordinate geometry.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

## Validation

The calculator accepts finite values only. Lengths, radii, and heights must be positive; triangles must satisfy the triangle inequality; regular polygons need at least three sides; and torus calculations use the ring-torus condition `R > r`.

Requires Python 3.13 or later and no third-party packages.
