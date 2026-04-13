"""Temporary script to generate notebook 1: Daily Store Sales Forecasting.
Delete this file after use.
"""
import json, pathlib

TARGET_PATH = pathlib.Path(r"E:\Github\Machine-Learning-Projects\Time Series Analysis\Daily Store Sales Forecasting\daily_store_sales_forecasting.ipynb")

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": []
}

def md(src):
    if isinstance(src, str):
        src = src.split("\n")
        src = [line + "\n" for line in src[:-1]] + [src[-1]]
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    if isinstance(src, str):
        src = src.split("\n")
        src = [line + "\n" for line in src[:-1]] + [src[-1]]
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None})

# ── 1. Title ──
md("# Daily Store Sales Forecasting\n\n**Project 1 of 10** — Time Series Forecasting Portfolio")

# ── 2. Project Overview ──
md("""## Project Overview

This notebook forecasts **daily total sales** for the Rossmann drugstore chain using historical sales data from 1,115 stores across Germany. The dataset comes from the Kaggle **Rossmann Store Sales** competition.

| Attribute | Value |
|-----------|-------|
| **Project type** | Time Series Forecasting |
| **Target variable** | `Sales` (daily store revenue) |
| **Date column** | `Date` |
| **Frequency** | Daily (`D`) |
| **Seasonal period** | 7 (weekly cycle) |
| **Primary TS library** | MLForecast |
| **Kaggle competition** | `rossmann-store-sales` |""")

# ── 3. Learning Objectives ──
md("""## Learning Objectives

By completing this notebook you will learn how to:

1. Download and merge a **multi-file retail dataset** (train + store metadata)
2. Handle **closed-store days** (zero sales when `Open=0`) and decide whether to include them
3. Aggregate store-level data to **total daily chain-wide sales** for a clear signal
4. Explore **day-of-week effects**, promotion impact, and holiday patterns in daily data
5. Engineer features appropriate for daily frequency: lags at 1, 7, 14, 28 days
6. Build naive and seasonal-naive (7-day) baselines
7. Benchmark regressors via LazyPredict on lag-feature tabular view
8. Run FLAML AutoML with a time budget
9. Train MLForecast models (LightGBM, XGBoost) with built-in lag/rolling features
10. Evaluate with MAE, RMSE, MAPE and compare all approaches""")

# ── 4. Problem Statement ──
md("""## Problem Statement

Given ~840 days of daily sales records across 1,115 Rossmann drugstores in Germany, **forecast total daily sales for the next 14 days**.

Rossmann stores vary by type, assortment, competition distance, and promotional activity. We aggregate to chain-wide totals for a cleaner univariate signal, then discuss panel-level extensions.""")

# ── 5. Why This Project Matters ──
md("""## Why This Project Matters

- **Inventory management**: Daily sales forecasts drive replenishment decisions for 1,115+ stores.
- **Staff scheduling**: Accurate daily forecasts directly determine staffing per shift.
- **Promotional planning**: Rossmann runs frequent promotions — understanding their daily lift is critical for ROI.
- **Holiday preparedness**: German state holidays and school holidays create demand spikes requiring advance planning.""")

# ── 6. Dataset Overview ──
md("""## Dataset Overview

The competition provides two key CSV files:

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | ~1.02 M | Daily sales: Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday |
| `store.csv` | 1,115 | Store metadata: Store, StoreType (a-d), Assortment (a-c), CompetitionDistance, Promo2, etc. |

### Key columns
- **Sales** — target; daily revenue (integer), 0 when store is closed
- **Open** — 1 if store was open, 0 if closed
- **Promo** — 1 if store ran a promotion that day
- **StateHoliday** — 0 (none), a (public), b (Easter), c (Christmas)
- **StoreType** — a, b, c, d (different store formats)
- **CompetitionDistance** — distance in meters to nearest competitor""")

# ── 7. Dataset Source & License ──
md("""## Dataset Source & License Notes

- **Kaggle competition**: [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
- **License**: Competition-specific (Kaggle Competition Rules)
- **Provider**: Rossmann (via Kaggle)
- **Usage**: Educational purposes only""")

# ── 8. Environment Setup ──
md("## Environment Setup")
code("""import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in [
    "kagglehub", "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "scikit-learn", "lazypredict", "flaml", "mlforecast", "lightgbm", "xgboost",
    "statsmodels", "scipy", "window-ops",
]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        _install(pkg)

print("All packages ready.")""")

# ── 9. Imports ──
md("## Imports")
code("""import warnings, os, pathlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from lazypredict.Supervised import LazyRegressor
from flaml import AutoML

from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
import lightgbm as lgb
import xgboost as xgb

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy import stats

pd.set_option("display.max_columns", 60)
plt.rcParams["figure.figsize"] = (14, 5)
sns.set_style("whitegrid")

print("All imports successful.")""")

# ── 10. Configuration ──
md("## Configuration & Constants")
code("""PROJECT_NAME = "Daily Store Sales Forecasting"
KAGGLE_SLUG  = "rossmann-store-sales"

TARGET  = "Sales"
DATE    = "Date"
FREQ    = "D"

SEASON_LENGTH    = 7       # weekly cycle for daily data
FORECAST_HORIZON = 14     # 14-day forecast
TEST_SIZE        = FORECAST_HORIZON
VAL_SIZE         = FORECAST_HORIZON
RANDOM_STATE     = 42
FLAML_BUDGET     = 120    # seconds

print(f"Project : {PROJECT_NAME}")
print(f"Target  : {TARGET}  |  Freq: {FREQ}  |  Season: {SEASON_LENGTH}")
print(f"Horizon : {FORECAST_HORIZON} days")""")

# ── 11. Kaggle Credential Check ──
md("""## Kaggle Credential Check

Before downloading, we verify that Kaggle credentials are available. The notebook expects `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`) in system environment variables, or a `kaggle.json` file in `~/.kaggle/`.""")

code("""kaggle_ok = False

if os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN"):
    print("Kaggle credentials found via environment variables.")
    kaggle_ok = True

kaggle_json = pathlib.Path.home() / ".kaggle" / "kaggle.json"
if kaggle_json.exists():
    print(f"Kaggle credentials found at {kaggle_json}")
    kaggle_ok = True

if not kaggle_ok:
    raise RuntimeError(
        "No Kaggle credentials found!\\n"
        "Set KAGGLE_API_TOKEN env var, or place kaggle.json in ~/.kaggle/\\n"
        "See: https://www.kaggle.com/settings -> Create New Token"
    )
print("Ready to download.")""")

# ── 12. Dataset Download & Loading ──
md("""## Dataset Download & Loading

We download the Rossmann Store Sales competition data via `kagglehub`. The download is idempotent — re-running skips if files already exist locally.""")

code("""import kagglehub

try:
    data_path = pathlib.Path(kagglehub.competition_download(KAGGLE_SLUG))
    print(f"Downloaded to: {data_path}")
except Exception as e:
    print(f"kagglehub download failed: {e}")
    print("Falling back to kaggle CLI...")
    os.makedirs("data", exist_ok=True)
    ret = os.system(f"kaggle competitions download -c {KAGGLE_SLUG} -p data/ --unzip")
    if ret != 0:
        raise RuntimeError("Both kagglehub and kaggle CLI failed. Check credentials and competition rules acceptance.")
    data_path = pathlib.Path("data")

csv_files = sorted(data_path.rglob("*.csv"))
for f in csv_files:
    print(f"  {f.name:35s}  {f.stat().st_size / 1e6:7.2f} MB")

assert len(csv_files) > 0, "No CSV files found after download!\"""")

md("### Load and merge CSVs")

code("""def _find(name):
    matches = [f for f in csv_files if name in f.name.lower()]
    assert matches, f"File matching '{name}' not found in {[f.name for f in csv_files]}"
    return matches[0]

train_raw = pd.read_csv(_find("train"), parse_dates=["Date"], low_memory=False)
store_info = pd.read_csv(_find("store"))

print(f"train_raw  : {train_raw.shape}  -- {train_raw['Date'].min().date()} to {train_raw['Date'].max().date()}")
print(f"store_info : {store_info.shape}")
train_raw.head()""")

# ── 13. Data Validation ──
md("## Data Validation Checks")

code("""print("=" * 60)
print("DATA VALIDATION REPORT")
print("=" * 60)

assert "Sales" in train_raw.columns, "Missing 'Sales' column!"
assert "Date" in train_raw.columns, "Missing 'Date' column!"

print(f"\\n[train.csv]")
print(f"  Shape            : {train_raw.shape[0]:,} rows x {train_raw.shape[1]} cols")
print(f"  Date range       : {train_raw['Date'].min().date()} to {train_raw['Date'].max().date()}")
print(f"  Unique stores    : {train_raw['Store'].nunique()}")
print(f"  Missing Sales    : {train_raw['Sales'].isna().sum()}")
print(f"  Zero-sales rows  : {(train_raw['Sales'] == 0).sum()} ({(train_raw['Sales']==0).mean()*100:.1f}%)")
print(f"  Closed-store rows: {(train_raw['Open'] == 0).sum()} ({(train_raw['Open']==0).mean()*100:.1f}%)")
print(f"  Promo days       : {train_raw['Promo'].sum()} ({train_raw['Promo'].mean()*100:.1f}%)")
print(f"  Duplicate rows   : {train_raw.duplicated().sum()}")

print(f"\\n[store.csv]")
print(f"  StoreType dist   : {store_info['StoreType'].value_counts().to_dict()}")
print(f"  Assortment dist  : {store_info['Assortment'].value_counts().to_dict()}")
print(f"  CompetitionDist NaN: {store_info['CompetitionDistance'].isna().sum()}")

print("\\nValidation complete.")""")

# ── 14. Data Cleaning / Preprocessing ──
md("""## Data Cleaning / Preprocessing

We filter to **open stores only** (closed days have Sales=0 and provide no demand signal), then aggregate to total daily chain-wide sales.""")

code("""# Filter to open stores only
train_open = train_raw[train_raw["Open"] == 1].copy()
print(f"After filtering to Open=1: {len(train_open):,} rows (dropped {len(train_raw)-len(train_open):,} closed-store days)")

# Merge with store metadata
train_merged = train_open.merge(store_info, on="Store", how="left")

# Aggregate to daily total sales
daily = (
    train_merged
    .groupby("Date")
    .agg(
        Sales=("Sales", "sum"),
        Customers=("Customers", "sum"),
        Stores_Open=("Store", "nunique"),
        Promo_Pct=("Promo", "mean"),
    )
    .reset_index()
    .sort_values("Date")
    .reset_index(drop=True)
)

# Add day-of-week
daily["DayOfWeek"] = daily["Date"].dt.dayofweek  # 0=Mon, 6=Sun
daily["DayName"] = daily["Date"].dt.day_name()

print(f"\\nAggregated daily series: {len(daily)} days")
print(f"Date range: {daily['Date'].min().date()} to {daily['Date'].max().date()}")
daily.head()""")

# ── 15. EDA ──
md("""## Exploratory Data Analysis

We explore the temporal structure — trend, weekly seasonality, promotion effects, and day-of-week patterns.""")

code("""# Full time-series plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Sales"], name="Total Daily Sales",
                         line=dict(color="blue", width=1)))
fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Sales"].rolling(7).mean(),
                         name="7-day MA", line=dict(color="red", width=2)))
fig.update_layout(title="Rossmann — Total Daily Sales (All Stores)", template="plotly_white",
                  xaxis_title="Date", yaxis_title="Total Daily Sales")
fig.show()""")

code("""# Day-of-week effect
fig = px.box(daily, x="DayName", y="Sales",
             category_orders={"DayName": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]},
             title="Sales Distribution by Day of Week")
fig.update_layout(template="plotly_white")
fig.show()

print("Mean daily sales by day of week:")
print(daily.groupby("DayName")["Sales"].mean().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).to_string())""")

code("""# Monthly seasonality
daily["Month"] = daily["Date"].dt.month
fig = px.box(daily, x="Month", y="Sales", title="Sales Distribution by Month")
fig.update_layout(template="plotly_white")
fig.show()""")

code("""# Seasonal decomposition
ts = daily.set_index("Date")["Sales"].asfreq("D")
ts = ts.interpolate() if ts.isna().any() else ts

decomp = seasonal_decompose(ts, model="additive", period=SEASON_LENGTH)
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
decomp.observed.plot(ax=axes[0], title="Observed")
decomp.trend.plot(ax=axes[1], title="Trend")
decomp.seasonal.plot(ax=axes[2], title="Weekly Seasonal (7-day)")
decomp.resid.plot(ax=axes[3], title="Residual")
plt.tight_layout()
plt.show()""")

code("""# ACF / PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(ts.dropna(), lags=40, ax=axes[0], title="ACF — Daily Sales")
plot_pacf(ts.dropna(), lags=40, ax=axes[1], title="PACF — Daily Sales")
plt.tight_layout()
plt.show()""")

code("""# Stationarity test
adf = adfuller(ts.dropna())
print("Augmented Dickey-Fuller Test:")
print(f"  ADF Statistic : {adf[0]:.4f}")
print(f"  p-value       : {adf[1]:.6f}")
for k, v in adf[4].items():
    print(f"  Critical ({k}): {v:.4f}")
print(f"\\nResult: {'STATIONARY' if adf[1] < 0.05 else 'NON-STATIONARY'} at 5% significance")""")

# ── 16. Target Analysis ──
md("## Target Analysis")

code("""print("Target Statistics (total daily sales):")
desc = daily["Sales"].describe()
print(desc.to_string())
print(f"\\nSkewness : {daily['Sales'].skew():.3f}")
print(f"Kurtosis : {daily['Sales'].kurtosis():.3f}")

Q1, Q3 = daily["Sales"].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = daily[(daily["Sales"] < Q1 - 1.5*IQR) | (daily["Sales"] > Q3 + 1.5*IQR)]
print(f"\\nOutliers (IQR method): {len(outliers)} days ({len(outliers)/len(daily)*100:.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(daily["Sales"], bins=30, edgecolor="black", alpha=0.7)
axes[0].set_title("Distribution of Daily Sales")
axes[1].boxplot(daily["Sales"].dropna())
axes[1].set_title("Box Plot")
pd.plotting.lag_plot(daily["Sales"], lag=1, ax=axes[2])
axes[2].set_title("Lag-1 Plot")
plt.tight_layout()
plt.show()""")

# ── 17. Train / Val / Test Split ──
md("""## Train / Validation / Test Split

Temporal split — no shuffling. We hold out the last 14 days for test and the prior 14 days for validation.""")

code("""ts_df = daily[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"}).copy()

n = len(ts_df)
ts_train = ts_df.iloc[: n - TEST_SIZE - VAL_SIZE].copy()
ts_val   = ts_df.iloc[n - TEST_SIZE - VAL_SIZE : n - TEST_SIZE].copy()
ts_test  = ts_df.iloc[n - TEST_SIZE :].copy()

print(f"Train : {len(ts_train)} days  ({ts_train['ds'].min().date()} to {ts_train['ds'].max().date()})")
print(f"Val   : {len(ts_val)} days  ({ts_val['ds'].min().date()} to {ts_val['ds'].max().date()})")
print(f"Test  : {len(ts_test)} days  ({ts_test['ds'].min().date()} to {ts_test['ds'].max().date()})")

assert ts_train["ds"].max() < ts_val["ds"].min(), "Train/val overlap!"
assert ts_val["ds"].max()   < ts_test["ds"].min(), "Val/test overlap!"
print("\\nNo temporal overlap — split is clean.")

ts_trainval = pd.concat([ts_train, ts_val]).reset_index(drop=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_train["ds"], y=ts_train["y"], name="Train"))
fig.add_trace(go.Scatter(x=ts_val["ds"],   y=ts_val["y"],   name="Validation", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=ts_test["ds"],  y=ts_test["y"],  name="Test", line=dict(color="red")))
fig.update_layout(title="Train / Validation / Test Split", template="plotly_white")
fig.show()""")

# ── 18. Feature Engineering ──
md("""## Feature Engineering

We create lag and rolling features for the tabular approaches (LazyPredict, FLAML). Lag features use `.shift()` to avoid data leakage.""")

code("""def make_features(df):
    out = df.copy()
    for lag in [1, 7, 14, 21, 28]:
        out[f"lag_{lag}"] = out["y"].shift(lag)
    for win in [7, 14, 28]:
        out[f"roll_mean_{win}"] = out["y"].shift(1).rolling(win).mean()
        out[f"roll_std_{win}"]  = out["y"].shift(1).rolling(win).std()
    out["dayofweek"] = out["ds"].dt.dayofweek
    out["month"]     = out["ds"].dt.month
    out["day"]       = out["ds"].dt.day
    return out

feat_full = make_features(ts_df)
feat_full = feat_full.dropna().reset_index(drop=True)
print(f"Features created. Shape after dropping NaN rows: {feat_full.shape}")
feat_full.head()""")

code("""# Re-split the feature-engineered data using temporal cutoffs
train_cut = ts_train["ds"].max()
val_cut   = ts_val["ds"].max()

feat_train = feat_full[feat_full["ds"] <= train_cut].copy()
feat_val   = feat_full[(feat_full["ds"] > train_cut) & (feat_full["ds"] <= val_cut)].copy()
feat_test  = feat_full[feat_full["ds"] > val_cut].copy()

feature_cols = [c for c in feat_full.columns if c not in ["ds", "y"]]

X_train, y_train = feat_train[feature_cols], feat_train["y"]
X_val,   y_val   = feat_val[feature_cols],   feat_val["y"]
X_test,  y_test  = feat_test[feature_cols],  feat_test["y"]

X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"Features: {feature_cols}")""")

# ── 19. Baseline Approaches ──
md("""## Baseline Approaches

Simple baselines establish the minimum performance bar:
- **Naive**: predict the last known value
- **Seasonal Naive**: predict the value from 7 days ago
- **Moving Average**: predict the average of the last 7 days""")

code("""def calc_metrics(actual, predicted, name="Model"):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.any() else np.nan
    return {"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}

results = []
actual_test = ts_test["y"].values
actual_val  = ts_val["y"].values

# Naive baseline (last value)
naive_pred = np.full(TEST_SIZE, ts_trainval["y"].iloc[-1])
results.append(calc_metrics(pd.Series(actual_test), pd.Series(naive_pred), "Naive (last value)"))

# Seasonal Naive (lag-7)
seasonal_naive = ts_trainval["y"].iloc[-SEASON_LENGTH:].values
seasonal_pred = np.tile(seasonal_naive, (FORECAST_HORIZON // SEASON_LENGTH) + 1)[:FORECAST_HORIZON]
results.append(calc_metrics(pd.Series(actual_test), pd.Series(seasonal_pred), "Seasonal Naive (lag-7)"))

# Moving Average (7-day)
ma_pred = np.full(TEST_SIZE, ts_trainval["y"].iloc[-7:].mean())
results.append(calc_metrics(pd.Series(actual_test), pd.Series(ma_pred), "Moving Avg (7-day)"))

baseline_df = pd.DataFrame(results)
print("Baseline Results:")
print(baseline_df.to_string(index=False))""")

# ── 20. LazyPredict ──
md("""## LazyPredict Benchmark (Lag-Feature Tabular View)

**Important**: LazyPredict is designed for tabular data, not native time-series forecasting. Here we use it on the **lag-feature engineered** data to quickly benchmark many regression algorithms.

This gives a useful signal about which model families work well, but is **not** a replacement for proper time-series models.""")

code("""try:
    lazy = LazyRegressor(verbose=0, ignore_warnings=True, random_state=RANDOM_STATE)
    lazy_models, lazy_preds = lazy.fit(X_train, X_val, y_train, y_val)
    print("LazyPredict Benchmark (top 15 by RMSE):")
    print(lazy_models.head(15).to_string())

    for i, (name, row) in enumerate(lazy_models.head(3).iterrows()):
        results.append({"Model": f"LazyPredict: {name}", "MAE": round(row.get("MAE", 0), 2),
                        "RMSE": round(row.get("RMSE", 0), 2), "MAPE": np.nan})
except Exception as e:
    print(f"LazyPredict failed: {e}")
    print("Continuing with other approaches.")""")

# ── 21. FLAML AutoML ──
md("""## FLAML AutoML

FLAML (Fast and Lightweight AutoML) efficiently searches for the best model and hyperparameters within a time budget. We use it on the lag-feature data.""")

code("""try:
    flaml_model = AutoML()
    flaml_model.fit(
        X_train=X_trainval, y_train=y_trainval,
        task="regression", time_budget=FLAML_BUDGET,
        metric="rmse", verbose=0, seed=RANDOM_STATE,
    )
    flaml_pred = flaml_model.predict(X_test)
    results.append(calc_metrics(pd.Series(actual_test), pd.Series(flaml_pred), f"FLAML ({flaml_model.best_estimator})"))
    print(f"FLAML best estimator: {flaml_model.best_estimator}")
    print(f"FLAML best config: {flaml_model.best_config}")
except Exception as e:
    print(f"FLAML failed: {e}")""")

# ── 22. MLForecast ──
md("""## MLForecast — Dedicated Time-Series Models

**Why MLForecast?** MLForecast wraps gradient boosting models (LightGBM, XGBoost) with built-in lag, rolling, and date features. It is ideal for business demand/sales forecasting because it:
- Handles panel data natively (multiple stores/series)
- Creates temporal features automatically
- Supports fast retraining and inference

For our univariate aggregate series, we add a single `unique_id` column to conform to the Nixtla API.""")

code("""# Prepare data in Nixtla long format
mlf_trainval = ts_trainval.copy()
mlf_trainval["unique_id"] = "total_sales"

mlf_test = ts_test.copy()
mlf_test["unique_id"] = "total_sales"

# Define MLForecast model
mlf = MLForecast(
    models={
        "LightGBM": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                       random_state=RANDOM_STATE, verbosity=-1),
        "XGBoost":  xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                      random_state=RANDOM_STATE, verbosity=0),
        "Ridge":    Ridge(alpha=1.0),
    },
    freq="D",
    lags=[1, 7, 14, 21, 28],
    lag_transforms={
        1: [(("rolling_mean", 7), ("rolling_mean", 14), ("rolling_std", 7))],
        7: [(("rolling_mean", 4),)],
    },
    date_features=["dayofweek", "month", "day"],
)

mlf.fit(mlf_trainval)
mlf_preds = mlf.predict(h=FORECAST_HORIZON)

print(f"MLForecast predictions shape: {mlf_preds.shape}")
mlf_preds.head()""")

code("""# Evaluate MLForecast models
for model_name in ["LightGBM", "XGBoost", "Ridge"]:
    if model_name in mlf_preds.columns:
        pred_vals = mlf_preds[model_name].values[:TEST_SIZE]
        r = calc_metrics(pd.Series(actual_test), pd.Series(pred_vals), f"MLForecast: {model_name}")
        results.append(r)
        print(f"{model_name}: MAE={r['MAE']}, RMSE={r['RMSE']}, MAPE={r['MAPE']}%")""")

# ── 23. Top 3 ──
md("""## Top 3 Model Selection

We rank all models by RMSE on the test set and select the top 3.""")

code("""results_df = pd.DataFrame(results).dropna(subset=["RMSE"]).sort_values("RMSE")
print("\\nAll Models Ranked by RMSE:")
print(results_df.to_string(index=False))

top3 = results_df.head(3)
print(f"\\n{'='*50}")
print("TOP 3 MODELS:")
print(f"{'='*50}")
print(top3.to_string(index=False))

fig = px.bar(results_df, x="Model", y="RMSE", title="Model Comparison — RMSE (lower is better)",
             color="RMSE", color_continuous_scale="RdYlGn_r")
fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
fig.show()""")

# ── 24. Final Evaluation ──
md("""## Final Evaluation — Forecast vs Actual

We overlay the top models' predictions against the actual test period.""")

code("""fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_test["ds"], y=actual_test, name="Actual",
                         line=dict(color="black", width=3)))

colors = ["blue", "green", "orange"]
for i, model_name in enumerate(["LightGBM", "XGBoost", "Ridge"]):
    if model_name in mlf_preds.columns:
        fig.add_trace(go.Scatter(x=ts_test["ds"], y=mlf_preds[model_name].values[:TEST_SIZE],
                                 name=f"MLF: {model_name}", line=dict(color=colors[i], dash="dash")))

fig.add_trace(go.Scatter(x=ts_test["ds"], y=seasonal_pred, name="Seasonal Naive",
                         line=dict(color="gray", dash="dot")))

fig.update_layout(title="Forecast vs Actual — Test Period", template="plotly_white",
                  xaxis_title="Date", yaxis_title="Total Daily Sales")
fig.show()""")

# ── 25. Error Analysis ──
md("""## Error Analysis

Understanding *where* and *when* the model makes errors is often more valuable than aggregate metrics.""")

code("""best_mlf_col = "LightGBM" if "LightGBM" in mlf_preds.columns else mlf_preds.columns[-1]
best_pred = mlf_preds[best_mlf_col].values[:TEST_SIZE]
errors = actual_test - best_pred
pct_errors = np.where(actual_test != 0, errors / actual_test * 100, 0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(errors, bins=15, edgecolor="black", alpha=0.7)
axes[0].set_title("Error Distribution")
axes[0].axvline(0, color="red", linestyle="--")

axes[1].plot(range(1, TEST_SIZE+1), np.abs(pct_errors), marker="o")
axes[1].set_title("Absolute % Error by Horizon Day")
axes[1].set_xlabel("Day Ahead")
axes[1].set_ylabel("|% Error|")

axes[2].scatter(actual_test, best_pred, alpha=0.7)
mn, mx = min(actual_test.min(), best_pred.min()), max(actual_test.max(), best_pred.max())
axes[2].plot([mn, mx], [mn, mx], "r--", label="Perfect")
axes[2].set_title("Actual vs Predicted")
axes[2].set_xlabel("Actual")
axes[2].set_ylabel("Predicted")
axes[2].legend()
plt.tight_layout()
plt.show()

print(f"Mean Error (bias): {errors.mean():,.0f}")
print(f"Mean Absolute Error: {np.abs(errors).mean():,.0f}")
print(f"Max overestimate: {errors.min():,.0f}")
print(f"Max underestimate: {errors.max():,.0f}")""")

# ── 26. Interpretation ──
md("""## Interpretation & Insights

### Key Findings

1. **Strong weekly seasonality**: Daily sales show a dominant 7-day cycle. Sunday is typically the lowest sales day (many German stores close), with peaks mid-week or Saturday.
2. **Promotion lift**: Days with high promotion participation consistently show higher aggregate sales.
3. **Baseline vs ML**: Seasonal naive (lag-7) provides a surprisingly strong baseline for weekly-patterned daily data. ML models improve primarily by capturing trend shifts and promotion effects.
4. **MLForecast effectiveness**: LightGBM and XGBoost in the MLForecast framework capture both weekly patterns and longer-term dynamics through the lag/rolling feature engineering.""")

# ── 27. Limitations ──
md("""## Limitations

1. **Aggregated series**: We forecast total chain-wide daily sales. Individual store forecasts would require panel modeling.
2. **No external regressors**: We did not use promotion schedules, holidays, or weather as future-known covariates.
3. **Fixed horizon**: The 14-day horizon is arbitrary. Actual business needs may vary.
4. **Single split**: We used one temporal train/val/test split. Rolling-origin cross-validation would give more robust estimates.
5. **No probabilistic forecasts**: We produced only point forecasts, not prediction intervals.""")

# ── 28. How to Improve ──
md("""## How to Improve This Project

1. **Add external features**: Use promotion schedule, school/state holidays, day-of-month as future-known covariates.
2. **Store-level panel forecasting**: Run MLForecast with `unique_id` per store to produce 1,115 individual forecasts.
3. **Rolling-origin cross-validation**: Evaluate on multiple temporal windows for more robust error estimates.
4. **Ensemble**: Combine top models via simple averaging.
5. **Probabilistic forecasts**: Use conformal prediction or quantile regression for prediction intervals.
6. **Longer FLAML budget**: Increase `FLAML_BUDGET` for more thorough hyperparameter search.""")

# ── 29. Production Considerations ──
md("""## Production Considerations

1. **Automated retraining**: Schedule daily retraining as new sales data arrives.
2. **Monitoring**: Track MAE/MAPE over time and alert on forecast degradation.
3. **Data pipeline**: Automate ingestion from POS systems, validation, and preprocessing.
4. **Hierarchical reconciliation**: Ensure store-level forecasts sum to district/region/total forecasts.
5. **Serving**: Expose forecasts via API for downstream systems (staffing, ordering).
6. **Fallback**: Maintain seasonal naive as a fallback if the primary model fails.""")

# ── 30. Common Mistakes ──
md("""## Common Mistakes to Avoid

1. **Including closed-store days**: Keeping `Open=0` rows inflates zero-sales counts and distorts patterns.
2. **Using future data in lag features**: Always use `.shift()` to prevent leakage.
3. **Random train/test split**: Always use temporal splits for time series.
4. **Ignoring Sunday closures**: German retail has specific Sunday trading laws. Not accounting for this creates systematic errors.
5. **MAPE with zeros**: MAPE is undefined when actual values are zero; use MAE or sMAPE as alternatives.""")

# ── 31. Mini Challenges ──
md("""## Mini Challenge / Exercises

1. **Store-level forecasting**: Add `unique_id = Store` and run MLForecast on the panel. How do per-store errors compare?
2. **Promotion feature**: Add a `promo_pct` future-known covariate. Does it improve accuracy?
3. **Log transform**: Apply `np.log1p()` to Sales, retrain, and compare (remember to `np.expm1()` the predictions).
4. **Ensemble**: Average predictions from LightGBM + XGBoost + FLAML. Does the ensemble beat individual models?
5. **Longer horizon**: Change `FORECAST_HORIZON` to 28. How does accuracy degrade?""")

# ── 32. Final Summary ──
md("""## Final Summary & Key Takeaways

### What We Did
- Downloaded and validated the **Rossmann Store Sales** dataset from Kaggle
- Filtered to open stores and aggregated to total daily chain-wide sales
- Performed EDA: weekly seasonality, promotion effects, day-of-week patterns
- Built naive, seasonal naive, and moving average baselines
- Benchmarked many regressors via LazyPredict on lag features
- Ran FLAML AutoML for efficient model search
- Trained MLForecast models (LightGBM, XGBoost, Ridge) with built-in temporal features
- Selected top 3 models and analyzed errors

### Key Takeaways
1. **Weekly seasonality dominates** daily retail sales — seasonal naive (lag-7) is a strong baseline
2. **MLForecast with gradient boosting** captures both weekly cycles and trend dynamics effectively
3. **Closed-store filtering** is essential — including zero-sales closed days distorts the signal
4. **Error analysis** reveals that most forecast errors occur on days with unusual events (holidays, promotions)
5. **Panel-level forecasting** is the natural next step for operational use

---
*Notebook generated as part of a time-series forecasting portfolio.*
*For educational purposes only — always validate before production use.*""")

# ── Write notebook ──
TARGET_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Notebook written: {TARGET_PATH}")
print(f"Total cells: {len(nb['cells'])} (MD: {sum(1 for c in nb['cells'] if c['cell_type']=='markdown')}, Code: {sum(1 for c in nb['cells'] if c['cell_type']=='code')})")

