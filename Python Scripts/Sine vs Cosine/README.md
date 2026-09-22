# Sine vs Cosine

A small NumPy and Matplotlib example that plots sine and cosine waves from
`-2π` to `2π` using 256 evenly spaced samples.

## Run interactively

```powershell
uv sync
uv run python app.py
```

Close the Matplotlib window to exit.

## Save a plot

```powershell
uv run python app.py --output sine_vs_cosine.png
```

The chart includes labeled axes, a zero reference line, a grid, and a legend
that distinguishes the two curves.

## Dependencies

uv manages NumPy and Matplotlib in `pyproject.toml`; exact resolved versions
are recorded in `uv.lock`.
