# Wine Quality Prediction Analysis - Classification

## 1. Project Overview

This project implements a **Classification** pipeline for **Wine Quality Prediction Analysis - Classification**. The target variable is `quality`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `quality` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `winequality.csv`

**Files in project directory:**

- `winequality.csv`

**Standardized data path:** `data/wine_quality_prediction_analysis_-_classification/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Handle missing values (fillna)
- Log transformation
- Drop columns/rows
- Oversampling (SMOTE)
- Train/test split

**Evaluation metrics:**
- Cross-Validation Score

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load winequality.csv] --> B[Handle missing values]
    B[Handle missing values] --> C[Log transformation]
    C[Log transformation] --> D[Drop columns/rows]
    D[Drop columns/rows] --> E[Oversampling]
    E[Oversampling] --> F[Train/test split]
    F[Train/test split] --> G[LazyClassifier Benchmark]
    G[LazyClassifier Benchmark] --> H[PyCaret Classification]
    H[PyCaret Classification] --> I[Evaluate: Cross-Validation Score]
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
Wine Quality Prediction Analysis - Classification/
├── Wine Quality Prediction Analysis.ipynb
├── winequality.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `imbalanced-learn`
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

- Open `Wine Quality Prediction Analysis.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p052_*.py`:

```bash
python -m pytest tests/test_p052_*.py -v
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
