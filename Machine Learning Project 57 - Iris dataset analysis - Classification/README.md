# Iris dataset analysis - Classification

## 1. Project Overview

This project implements a **Classification** pipeline for **Iris dataset analysis - Classification**. The target variable is `Species`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `Species` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `Iris.csv`

**Files in project directory:**

- `Iris.csv`

**Standardized data path:** `data/iris_dataset_analysis_-_classification/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Label encoding (LabelEncoder)
- Train/test split

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Iris.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Label encoding]
    C[Label encoding] --> D[Train/test split]
    D[Train/test split] --> E[LazyClassifier Benchmark]
    E[LazyClassifier Benchmark] --> F[PyCaret Classification]
    F[PyCaret Classification] --> G[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 47 |
| Code cells | 33 |
| Markdown cells | 14 |
| Original cells | 34 |
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
Iris dataset analysis - Classification/
├── Iris Dataset Analysis - Classification.ipynb
├── Iris.csv
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

- Open `Iris Dataset Analysis - Classification.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p057_*.py`:

```bash
python -m pytest tests/test_p057_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyClassifier** benchmark cell
- **PyCaret Classification** pipeline cell
