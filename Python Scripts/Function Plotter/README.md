# Function Plotter

Terminal ASCII plotter for mathematical expressions in `x`.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py --func "sin(x)" --xmin -6.28 --xmax 6.28
```

Repeat `--func` to draw more than one curve. Omit it to use interactive mode.

## Expression rules

Expressions allow numbers, `x`, arithmetic operators, and public functions or constants from Python's `math` module. Calls to attributes, imports, strings, comprehensions, and other Python syntax are rejected before evaluation.

The plotter requires Python 3.13 or later and has no third-party dependencies.
