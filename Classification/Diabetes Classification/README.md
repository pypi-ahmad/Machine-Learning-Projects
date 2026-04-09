# Diabetes Classification

## 1. Project Overview

This project implements a **Classification** pipeline for **Diabetes Classification**. The target variable is `Outcome`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `Outcome` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `kaggle_diabetes.csv`

**Standardized data path:** `data/diabetes_classification/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Handle missing values (fillna)
- Drop columns/rows
- Train/test split
- Feature scaling (StandardScaler)

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load kaggle_diabetes.csv] --> B[Handle missing values]
    B[Handle missing values] --> C[Drop columns/rows]
    C[Drop columns/rows] --> D[Train/test split]
    D[Train/test split] --> E[Feature scaling]
    E[Feature scaling] --> F[LazyClassifier Benchmark]
    F[LazyClassifier Benchmark] --> G[PyCaret Classification]
    G[PyCaret Classification] --> H[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 36 |
| Code cells | 26 |
| Markdown cells | 10 |
| Original cells | 23 |
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
Diabetes Classification/
├── Diabetes Classification.ipynb
├── dataset
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

- Open `Diabetes Classification.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p064_*.py`:

```bash
python -m pytest tests/test_p064_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyClassifier** benchmark cell
- **PyCaret Classification** pipeline cell
