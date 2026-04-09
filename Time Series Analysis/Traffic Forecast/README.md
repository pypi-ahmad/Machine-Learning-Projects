# Traffic Forecast - Time Series Analysis

## 1. Project Overview

This project implements a **Regression** pipeline for **Traffic Forecast - Time Series Analysis**. The target variable is `ID`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `ID` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `Traffic data.csv`

**Files in project directory:**

- `Traffic data.csv`

**Standardized data path:** `data/traffic_forecast_-_time_series_analysis/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Date parsing
- Drop columns/rows
- Train/test split

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Traffic data.csv] --> B[Date parsing]
    B[Date parsing] --> C[Drop columns/rows]
    C[Drop columns/rows] --> D[Train/test split]
    D[Train/test split] --> E[LazyRegressor Benchmark]
    E[LazyRegressor Benchmark] --> F[PyCaret Regression]
    F[PyCaret Regression] --> G[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 34 |
| Code cells | 21 |
| Markdown cells | 13 |
| Original cells | 21 |
| Standardized cells (added) | 13 |
| Original model training | None — preprocessing/EDA only |

## 6. Model Details

### LazyRegressor (Standardized)

Compares 20+ regressors, ranked by RMSE/R².

### PyCaret Regression (Standardized)

AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> ⚠️ Requires Python ≤ 3.11.

## 7. Project Structure

```
Traffic Forecast - Time Series Analysis/
├── Traffic Forecast - Time Series Analysis.ipynb
├── Traffic data.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `lazypredict`
- `matplotlib`
- `numpy`
- `pandas`
- `pycaret`
- `scikit-learn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Traffic Forecast - Time Series Analysis.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p054_*.py`:

```bash
python -m pytest tests/test_p054_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
