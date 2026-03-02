# Loan Prediction Analysis - Classification

## 1. Project Overview

This project implements a **Classification** pipeline for **Loan Prediction Analysis - Classification**. The target variable is `Loan_Status`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `Loan_Status` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `Loan Prediction Dataset.csv`

**Files in project directory:**

- `Loan Prediction Dataset.csv`

**Standardized data path:** `data/loan_prediction_analysis_-_classification/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Handle missing values (fillna)
- Log transformation
- Drop columns/rows
- Label encoding (LabelEncoder)
- Train/test split

**Evaluation metrics:**
- Cross-Validation Score

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Loan Prediction Dataset.csv] --> B[Handle missing values]
    B[Handle missing values] --> C[Log transformation]
    C[Log transformation] --> D[Drop columns/rows]
    D[Drop columns/rows] --> E[Label encoding]
    E[Label encoding] --> F[Train/test split]
    F[Train/test split] --> G[LazyClassifier Benchmark]
    G[LazyClassifier Benchmark] --> H[PyCaret Classification]
    H[PyCaret Classification] --> I[Evaluate: Cross-Validation Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 60 |
| Code cells | 43 |
| Markdown cells | 17 |
| Original cells | 47 |
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

### Evaluation Metrics

- Cross-Validation Score

## 7. Project Structure

```
Loan Prediction Analysis - Classification/
├── Loan Prediction Analysis - Classification.ipynb
├── Loan Prediction Dataset.csv
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

- Open `Loan Prediction Analysis - Classification.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p056_*.py`:

```bash
python -m pytest tests/test_p056_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyClassifier** benchmark cell
- **PyCaret Classification** pipeline cell
