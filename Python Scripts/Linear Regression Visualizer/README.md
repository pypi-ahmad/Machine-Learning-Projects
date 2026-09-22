# Linear Regression Visualizer

A standard-library CLI for fitting simple linear regression, reporting OLS coefficients and R-squared, and displaying an optional ASCII plot.

```powershell
uv sync --no-config
uv run --no-config python main.py --demo
```

For CSV input, provide the input file and numeric column names: `--file data.csv --x feature --y target`. Use `--noplot` to skip the ASCII visualization.
