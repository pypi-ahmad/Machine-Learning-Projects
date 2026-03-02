# Wine Quality Prediction

## 1. Project Overview

This project implements a **Classification** pipeline for **Wine Quality Prediction**. The target variable is `quality`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `quality` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `winequality-red.csv`

**Files in project directory:**

- `winequality-red.csv`

**Standardized data path:** `data/wine_quality_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Label encoding (LabelEncoder)
- Drop columns/rows
- Train/test split
- Feature scaling (StandardScaler)

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load winequality-red.csv] --> B[Label encoding]
    B[Label encoding] --> C[Drop columns/rows]
    C[Drop columns/rows] --> D[Train/test split]
    D[Train/test split] --> E[Feature scaling]
    E[Feature scaling] --> F[LazyClassifier Benchmark]
    F[LazyClassifier Benchmark] --> G[PyCaret Classification]
    G[PyCaret Classification] --> H[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 20 |
| Code cells | 14 |
| Markdown cells | 6 |
| Original cells | 7 |
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
Wine Quality Prediction/
├── prediction-of-quality-of-wine.ipynb
├── winequality-red.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `lazypredict`
- `matplotlib`
- `pycaret`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `prediction-of-quality-of-wine.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p043_*.py`:

```bash
python -m pytest tests/test_p043_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyClassifier** benchmark cell
- **PyCaret Classification** pipeline cell
