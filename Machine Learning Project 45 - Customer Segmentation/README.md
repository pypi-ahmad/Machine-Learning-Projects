# Customer Segmentation

## 1. Project Overview

This project implements a **Clustering** pipeline for **Customer Segmentation**.

| Property | Value |
|----------|-------|
| **ML Task** | Clustering |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `Customers.csv`
- `CC_GENERAL.csv`

**Files in project directory:**

- `Customers.csv`

**Standardized data path:** `data/customer_segmentation/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Feature scaling (MinMaxScaler)
- Handle missing values (fillna)
- Drop columns/rows

### Standardized Pipeline (added)

- **PyCaret Clustering**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Customers.csv] --> B[Feature scaling]
    B[Feature scaling] --> C[Handle missing values]
    C[Handle missing values] --> D[Drop columns/rows]
    D[Drop columns/rows] --> E[PyCaret Clustering]
    E[PyCaret Clustering] --> F[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 28 |
| Code cells | 22 |
| Markdown cells | 6 |
| Original cells | 16 |
| Standardized cells (added) | 12 |
| Original model training | None — preprocessing/EDA only |

## 6. Model Details

### PyCaret Clustering (Standardized)

AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> ⚠️ Requires Python ≤ 3.11.

### Evaluation Metrics

- Silhouette Score

## 7. Project Structure

```
Customer Segmentation/
├── K-Means.ipynb
├── Customers.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

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

- Open `K-Means.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p045_*.py`:

```bash
python -m pytest tests/test_p045_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **PyCaret Clustering** pipeline cell
