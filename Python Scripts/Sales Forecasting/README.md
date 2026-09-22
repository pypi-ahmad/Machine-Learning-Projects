# Sales Forecasting

A Streamlit learning demo that compares moving-average, exponential-smoothing,
and linear-trend sales forecasts. Upload a CSV or explore the repeatable sample
monthly series.

## Run

```powershell
uv sync
uv run streamlit run main.py
```

## Input CSV

Upload a CSV with one date column and one numeric sales column. Select the
matching columns in the app. At least three rows are required.

## Model comparison

The app uses an 80/20 chronological split. Moving-average and exponential-
smoothing test predictions are one-step-ahead: they use observations strictly
before the value being scored. The linear-trend forecast is fit on the training
period and projected across the full test horizon.

These simple methods are useful baselines, not production demand forecasts.
They do not account for promotions, stock-outs, holidays, or changing market
conditions.

## Dependencies

uv manages pandas and Streamlit in `pyproject.toml`; the resolved versions are
recorded in `uv.lock`.
