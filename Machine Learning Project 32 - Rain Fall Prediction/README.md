# Rain Fall Prediction

## 1. Project Overview

This project implements a **Classification** pipeline for **Rain Fall Prediction**. The target variable is `RainTomorrow`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `RainTomorrow` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `weatherAUS.csv`

**Files in project directory:**

- `weatherAUS.csv`

**Standardized data path:** `data/rain_fall_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Handle missing values (fillna)
- One-hot encoding (pd.get_dummies)
- Label mapping (function)
- Drop columns/rows
- Train/test split

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load weatherAUS.csv] --> B[Handle missing values]
    B[Handle missing values] --> C[One-hot encoding]
    C[One-hot encoding] --> D[Label mapping]
    D[Label mapping] --> E[Drop columns/rows]
    E[Drop columns/rows] --> F[Train/test split]
    F[Train/test split] --> G[LazyClassifier Benchmark]
    G[LazyClassifier Benchmark] --> H[PyCaret Classification]
    H[PyCaret Classification] --> I[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 23 |
| Code cells | 17 |
| Markdown cells | 6 |
| Original cells | 10 |
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
Rain Fall Prediction/
├── RainPrediction2.ipynb
├── weatherAUS.csv
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

- Open `RainPrediction2.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p032_*.py`:

```bash
python -m pytest tests/test_p032_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyClassifier** benchmark cell
- **PyCaret Classification** pipeline cell
