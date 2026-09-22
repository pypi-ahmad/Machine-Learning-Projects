# Statistics Calculator

A standard-library command-line calculator for descriptive statistics,
z-scores, percentiles, histograms, correlation, and simple linear regression.

## Run

```powershell
uv run python main.py
```

Enter numeric values separated by spaces or commas, then choose a menu option.
Use `0` to quit.

## Included calculations

- Descriptive statistics: size, range, mean, median, mode, quartiles, IQR,
  sample variance, sample standard deviation, skewness, and kurtosis.
- Z-scores and ASCII frequency histograms.
- Pearson correlation and least-squares linear regression.
- Interpolated percentile lookup from 0 through 100.

Constant datasets receive zero z-scores. Histograms require at least one bin,
and percentile values outside 0–100 are rejected.

## Limitations

This is an educational calculator, not a statistical inference package. The
hypothesis-testing claims from earlier versions are not present in the code, so
they are not documented here. Interpret correlation and regression in context;
they do not establish causation.

## Dependencies

The project uses only Python's standard library. uv records the Python
requirement and provides the reproducible environment.
