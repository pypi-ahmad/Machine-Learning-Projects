# Adult Salary Prediction

## 1. Project Overview

This project implements a **Classification** pipeline for **Adult Salary Prediction**. The target variable is `salary`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `salary` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `adult_data.csv`

**Files in project directory:**

- `adult_data.csv`

**Standardized data path:** `data/adult_salary_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Log transformation
- Outlier removal (IQR/quantile)
- Label mapping (function)
- Drop columns/rows
- Feature scaling (StandardScaler)
- Train/test split

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load adult_data.csv] --> B[Log transformation]
    B[Log transformation] --> C[Outlier removal]
    C[Outlier removal] --> D[Label mapping]
    D[Label mapping] --> E[Drop columns/rows]
    E[Drop columns/rows] --> F[Feature scaling]
    F[Feature scaling] --> G[Train/test split]
    G[Train/test split] --> H[LazyClassifier Benchmark]
    H[LazyClassifier Benchmark] --> I[PyCaret Classification]
    I[PyCaret Classification] --> J[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 63 |
| Code cells | 52 |
| Markdown cells | 11 |
| Original cells | 50 |
| Standardized cells (added) | 13 |
| Original model training | None — preprocessing/EDA only |

**⚠️ Deprecated APIs detected:**

- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`

## 6. Model Details

### LazyClassifier (Standardized)

Compares 20+ classifiers, ranked by accuracy/F1.

### PyCaret Classification (Standardized)

AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> ⚠️ Requires Python ≤ 3.11.

## 7. Project Structure

```
Adult Salary Prediction/
├── Untitled.ipynb
├── adult_data.csv
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

Automated tests are available in `tests/test_p001_*.py`:

```bash
python -m pytest tests/test_p001_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`
- No original model training exists — only auto-generated LazyPredict/PyCaret cells
- Notebook uses default name (`Untitled.ipynb`)

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyClassifier** benchmark cell
- **PyCaret Classification** pipeline cell
