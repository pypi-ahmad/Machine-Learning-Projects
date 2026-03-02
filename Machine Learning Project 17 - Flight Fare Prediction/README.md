# Flight Fare Prediction

## 1. Project Overview

This project implements a **Regression** pipeline for **Flight Fare Prediction**. The target variable is `Airline`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `Airline` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Files in project directory:**

- `Data_Train.xlsx`
- `Test_set.xlsx`

**Standardized data path:** `data/flight_fare_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop missing values (dropna)
- Date parsing
- Drop columns/rows
- One-hot encoding (pd.get_dummies)
- Value replacement (dict-based)
- DataFrame concatenation
- Train/test split

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Drop missing values]
    B[Drop missing values] --> C[Date parsing]
    C[Date parsing] --> D[Drop columns/rows]
    D[Drop columns/rows] --> E[One-hot encoding]
    E[One-hot encoding] --> F[Value replacement]
    F[Value replacement] --> G[DataFrame concatenation]
    G[DataFrame concatenation] --> H[Train/test split]
    H[Train/test split] --> I[LazyRegressor Benchmark]
    I[LazyRegressor Benchmark] --> J[PyCaret Regression]
    J[PyCaret Regression] --> K[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 79 |
| Code cells | 69 |
| Markdown cells | 10 |
| Original cells | 66 |
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
Flight Fare Prediction/
├── Untitled.ipynb
├── Data_Train.xlsx
├── Test_set.xlsx
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
- `seaborn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Untitled.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p017_*.py`:

```bash
python -m pytest tests/test_p017_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells
- Notebook uses default name (`Untitled.ipynb`)
- Hardcoded file paths detected — may need adjustment

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
