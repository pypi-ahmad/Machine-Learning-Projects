# Heart Disease Prediction

## 1. Project Overview

This project implements a **Classification** pipeline for **Heart Disease Prediction**. The target variable is `target`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `target` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `heart.csv`

**Files in project directory:**

- `heart.csv`

**Standardized data path:** `data/heart_disease_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- One-hot encoding (pd.get_dummies)
- Feature scaling (StandardScaler)
- Drop columns/rows

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load heart.csv] --> B[One-hot encoding]
    B[One-hot encoding] --> C[Feature scaling]
    C[Feature scaling] --> D[Drop columns/rows]
    D[Drop columns/rows] --> E[LazyClassifier Benchmark]
    E[LazyClassifier Benchmark] --> F[PyCaret Classification]
    F[PyCaret Classification] --> G[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 41 |
| Code cells | 26 |
| Markdown cells | 15 |
| Original cells | 28 |
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
Heart Disease Prediction/
├── Heart Disease Prediction.ipynb
├── heart.csv
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

- Open `Heart Disease Prediction.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p065_*.py`:

```bash
python -m pytest tests/test_p065_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyClassifier** benchmark cell
- **PyCaret Classification** pipeline cell
