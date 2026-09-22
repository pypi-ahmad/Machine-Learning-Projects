# Equation Solver

A dependency-free command-line calculator for linear, quadratic, and cubic equations, plus two-by-two linear systems.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup

```powershell
uv sync --no-config
```

## Examples

Solve a linear equation in `ax+b=c` form:

```powershell
uv run --no-config python main.py --linear "3x+2=11"
```

Solve a quadratic or cubic by entering its coefficients:

```powershell
uv run --no-config python main.py --quadratic 1 -3 2
uv run --no-config python main.py --cubic 1 0 -1 0
```

Solve `a1x+b1y=c1` and `a2x+b2y=c2`:

```powershell
uv run --no-config python main.py --system "2 1 5" "1 -1 1"
```

Run without arguments for the interactive prompts:

```powershell
uv run --no-config python main.py
```

## Notes

- Quadratic results include real or complex roots.
- Cubic results are numerical approximations of real roots; repeated roots or difficult polynomials may not be found reliably.
- A system with a zero determinant has no unique solution.
