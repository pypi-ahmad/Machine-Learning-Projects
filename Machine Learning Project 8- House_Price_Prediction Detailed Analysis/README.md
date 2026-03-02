# House_Price_Prediction Detailed Analysis

## 1. Project Overview

This project implements a **Regression** pipeline for **House_Price_Prediction Detailed Analysis**. The target variable is `None`.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Target Variable** | `None` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Standardized data path:** `data/house_price_prediction_detailed_analysis/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Log transformation

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Log transformation]
    B[Log transformation] --> C[LazyRegressor Benchmark]
    C[LazyRegressor Benchmark] --> D[PyCaret Regression]
    D[PyCaret Regression] --> E[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 112 |
| Code cells | 82 |
| Markdown cells | 30 |
| Original cells | 45 |
| Standardized cells (added) | 67 |
| Original model training | None — preprocessing/EDA only |

## 6. Model Details

### LazyRegressor (Standardized)

Compares 20+ regressors, ranked by RMSE/R².

### PyCaret Regression (Standardized)

AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> ⚠️ Requires Python ≤ 3.11.

## 7. Project Structure

```
Machine Learning Project 8- House_Price_Prediction Detailed Analysis/
├── Data Analysis.ipynb
├── Feature_Engineering.ipynb
├── Feature_Selection.ipynb
├── dataset
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

- Open `Data Analysis.ipynb` and run all cells
- Open `Feature_Engineering.ipynb` and run all cells
- Open `Feature_Selection.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p008_*.py`:

```bash
python -m pytest tests/test_p008_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells
- Hardcoded file paths detected — may need adjustment

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
