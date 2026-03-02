# Hand Digit Recognition Using ML

## 1. Project Overview

This project implements a **Regression** pipeline for **Hand Digit Recognition Using ML**.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Dataset Status** | BLOCKED KAGGLE |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `load_digits() (sklearn built-in)`

> ⚠️ **Dataset not available locally.** kaggle: dczerniawko/fifa19-analysis

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Feature scaling (MinMaxScaler)
- Train/test split

### Standardized Pipeline (added)

- **LazyRegressor**: Automated comparison of multiple models in a single call
- **PyCaret Regression**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load load_digits() (sklearn built-in)] --> B[Feature scaling]
    B[Feature scaling] --> C[Train/test split]
    C[Train/test split] --> D[LazyRegressor Benchmark]
    D[LazyRegressor Benchmark] --> E[PyCaret Regression]
    E[PyCaret Regression] --> F[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 17 |
| Code cells | 11 |
| Markdown cells | 6 |
| Original cells | 4 |
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
Hand Digit Recognition Using ML/
├── Untitled.ipynb
├── FIFA Data Analysis
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `lazypredict`
- `matplotlib`
- `pandas`
- `pycaret`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Untitled.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p012_*.py`:

```bash
python -m pytest tests/test_p012_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells
- Notebook uses default name (`Untitled.ipynb`)

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyRegressor** benchmark cell
- **PyCaret Regression** pipeline cell
