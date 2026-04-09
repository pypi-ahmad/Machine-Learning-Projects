# Employee Turnover Prediction

## 1. Project Overview

This project implements a **Classification** pipeline for **Employee Turnover Prediction**. The target variable is `department`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `department` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `dataset.csv`

**Files in project directory:**

- `dataset.csv`

**Standardized data path:** `data/employee_turnover_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- One-hot encoding (pd.get_dummies)
- Drop columns/rows
- Feature scaling (StandardScaler)
- Train/test split

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load dataset.csv] --> B[One-hot encoding]
    B[One-hot encoding] --> C[Drop columns/rows]
    C[Drop columns/rows] --> D[Feature scaling]
    D[Feature scaling] --> E[Train/test split]
    E[Train/test split] --> F[LazyClassifier Benchmark]
    F[LazyClassifier Benchmark] --> G[PyCaret Classification]
    G[PyCaret Classification] --> H[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 39 |
| Code cells | 32 |
| Markdown cells | 7 |
| Original cells | 26 |
| Standardized cells (added) | 13 |
| Original model training | None — preprocessing/EDA only |

## 6. Model Details

### LazyClassifier (Standardized)

Compares 20+ classifiers, ranked by accuracy/F1.

### PyCaret Classification (Standardized)

AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> ⚠️ Requires Python ≤ 3.11.

## 7. Project Structure

```
Employee Turnover Prediction/
├── Untitled.ipynb
├── dataset.csv
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

Automated tests are available in `tests/test_p022_*.py`:

```bash
python -m pytest tests/test_p022_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells
- Notebook uses default name (`Untitled.ipynb`)

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyClassifier** benchmark cell
- **PyCaret Classification** pipeline cell
