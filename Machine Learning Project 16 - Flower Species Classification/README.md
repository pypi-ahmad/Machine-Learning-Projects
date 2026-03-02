# Flower Species Classification

## 1. Project Overview

This project implements a **Classification** pipeline for **Flower Species Classification**.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Dataset Status** | OK BUILTIN |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `seaborn.load_dataset('iris')`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Label mapping (function)
- Feature scaling (StandardScaler)
- Train/test split

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load seaborn.load_dataset('iris')] --> B[Label mapping]
    B[Label mapping] --> C[Feature scaling]
    C[Feature scaling] --> D[Train/test split]
    D[Train/test split] --> E[LazyClassifier Benchmark]
    E[LazyClassifier Benchmark] --> F[PyCaret Classification]
    F[PyCaret Classification] --> G[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 31 |
| Code cells | 25 |
| Markdown cells | 6 |
| Original cells | 18 |
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
Flower Species Classification/
├── Untitled.ipynb
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

Automated tests are available in `tests/test_p016_*.py`:

```bash
python -m pytest tests/test_p016_*.py -v
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
