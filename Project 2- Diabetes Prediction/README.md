# Diabetes Prediction

## 1. Project Overview

This project implements a **Classification** pipeline for **Diabetes Prediction**. The target variable is `Date`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `Date` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `diabetes.csv`

**Files in project directory:**

- `diabetes.csv`

**Standardized data path:** `data/diabetes_prediction/`

## 3. Pipeline Overview

The original notebook primarily contains data loading and exploratory data analysis.

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load diabetes.csv] --> B[LazyClassifier Benchmark]
    B[LazyClassifier Benchmark] --> C[PyCaret Classification]
    C[PyCaret Classification] --> D[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 43 |
| Code cells | 27 |
| Markdown cells | 16 |
| Original cells | 1 |
| Standardized cells (added) | 42 |
| Original model training | None — preprocessing/EDA only |

## 6. Model Details

### LazyClassifier (Standardized)

Compares 20+ classifiers, ranked by accuracy/F1.

### PyCaret Classification (Standardized)

AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> ⚠️ Requires Python ≤ 3.11.

### Evaluation Metrics

- Accuracy
- Classification Report
- Confusion Matrix

## 7. Project Structure

```
Project 2- Diabetes Prediction/
├── Diabetes Prediction using PySpark.ipynb
├── Diabetes Prediction.ipynb
├── diabetes.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `lazypredict`
- `matplotlib`
- `pandas`
- `pycaret`
- `scikit-learn`
- `seaborn`
- `xgboost`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Diabetes Prediction using PySpark.ipynb` and run all cells
- Open `Diabetes Prediction.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p002_*.py`:

```bash
python -m pytest tests/test_p002_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyClassifier** benchmark cell
- **PyCaret Classification** pipeline cell
