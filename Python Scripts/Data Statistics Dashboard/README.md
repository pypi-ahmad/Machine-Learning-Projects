# Data Statistics Dashboard

A local Streamlit dashboard for exploring a CSV or Excel workbook in the browser. It reports descriptive statistics, distributions, correlations, outliers, and basic data-quality counts.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run streamlit run main.py
```

Upload a `.csv`, `.xlsx`, or `.xls` file, then choose the columns to analyze.

## Privacy and limits

- Uploaded files are read in memory for the active Streamlit session; this project does not write them to disk or send them to a remote service.
- Results depend on the selected columns and the input data quality. Missing values, mixed types, small samples, and extreme values can make statistics misleading.
- Correlation measures association, not causation. The outlier views are simple IQR or z-score heuristics, not a domain-specific review.

## Project files

```text
main.py         # Streamlit application
pyproject.toml  # uv dependency definition
uv.lock         # Resolved dependency versions
```

## Verification

```powershell
uv run python -m py_compile main.py
uv lock --check
```
