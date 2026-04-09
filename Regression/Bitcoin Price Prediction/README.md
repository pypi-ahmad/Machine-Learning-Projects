# Bitcoin Price Prediction

## 1. Project Overview

This project implements a **Classification** pipeline for **Bitcoin Price Prediction**. The target variable is `Date`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `Date` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `bitcoin.csv`

**Files in project directory:**

- `bitcoin.csv`

**Standardized data path:** `data/bitcoin_price_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Train/test split

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load bitcoin.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Train/test split]
    C[Train/test split] --> D[LazyClassifier Benchmark]
    D[LazyClassifier Benchmark] --> E[PyCaret Classification]
    E[PyCaret Classification] --> F[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 17 |
| Code cells | 11 |
| Markdown cells | 6 |
| Original cells | 4 |
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
Bitcoin Price Prediction/
├── Untitled.ipynb
├── bitcoin.csv
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

- Open `Untitled.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p002_*.py`:

```bash
python -m pytest tests/test_p002_*.py -v
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
