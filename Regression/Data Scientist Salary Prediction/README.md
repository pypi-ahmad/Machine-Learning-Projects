# Data Scientist's Salary Prediction

## 1. Project Overview

This project implements a **Regression** pipeline for **Data Scientist's Salary Prediction**. The target variable is `salary`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `salary` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `glassdoor_jobs.csv`

**Files in project directory:**

- `glassdoor_jobs.csv`

**Standardized data path:** `data/data_scientists_salary_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Handle missing values (fillna)
- Data type conversion
- Lowercasing
- Label mapping (function)
- One-hot encoding (pd.get_dummies)
- Feature scaling (StandardScaler)

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load glassdoor_jobs.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Handle missing values]
    C[Handle missing values] --> D[Data type conversion]
    D[Data type conversion] --> E[Lowercasing]
    E[Lowercasing] --> F[Label mapping]
    F[Label mapping] --> G[One-hot encoding]
    G[One-hot encoding] --> H[Feature scaling]
    H[Feature scaling] --> I[LazyRegressor Benchmark]
    I[LazyRegressor Benchmark] --> J[PyCaret Regression]
    J[PyCaret Regression] --> K[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 112 |
| Code cells | 91 |
| Markdown cells | 21 |
| Original cells | 99 |
| Standardized cells (added) | 13 |
| Original model training | None — preprocessing/EDA only |

**⚠️ Deprecated APIs detected:**

- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`

## 6. Model Details

### LazyRegressor (Standardized)

Compares 20+ regressors, ranked by RMSE/R².

### PyCaret Regression (Standardized)

AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> ⚠️ Requires Python ≤ 3.11.

## 7. Project Structure

```
Data Scientist's Salary Prediction/
├── Data Scientist's Salary Prediction.ipynb
├── glassdoor_jobs.csv
├── readme-resources
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

- Open `Data Scientist's Salary Prediction.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p063_*.py`:

```bash
python -m pytest tests/test_p063_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
