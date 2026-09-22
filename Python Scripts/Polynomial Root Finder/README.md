# Polynomial Root Finder

Find real and complex roots of a polynomial with the Durand-Kerner method.

## Setup

```powershell
uv sync --no-config
```

## Run

Provide coefficients from the highest-degree term to the constant term:

```powershell
uv run --no-config python main.py --coeffs "1 -6 11 -6"
```

The example represents `x³ - 6x² + 11x - 6` and has roots 1, 2, and 3. Commas are also accepted:

```powershell
uv run --no-config python main.py --coeffs "1, 0, 1"
```

Run without `--coeffs` to enter polynomials interactively. Type `quit`, `q`, or `exit` to leave the prompt.

## Notes

The tool reports the maximum residual for the roots it finds. Numerical root-finding is approximate; repeated roots, ill-conditioned coefficients, and high-degree polynomials can reduce accuracy.

It runs locally with the Python standard library and makes no network requests.
