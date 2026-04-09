# Turkiye Student Evaluation Analysis - Clustering

## 1. Project Overview

This project implements a **Clustering** pipeline for **Turkiye Student Evaluation Analysis - Clustering**.

| Property | Value |
|----------|-------|
| **ML Task** | Clustering |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `turkiye-student-evaluation_generic.csv`

**Files in project directory:**

- `turkiye-student-evaluation_generic.csv`

**Standardized data path:** `data/turkiye_student_evaluation_analysis_-_clustering/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Index setting
- Dimensionality reduction (PCA)

### Standardized Pipeline (added)

- **PyCaret Clustering**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load turkiye-student-evaluation_generic.csv] --> B[Index setting]
    B[Index setting] --> C[Dimensionality reduction]
    C[Dimensionality reduction] --> D[PyCaret Clustering]
    D[PyCaret Clustering] --> E[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 37 |
| Code cells | 23 |
| Markdown cells | 14 |
| Original cells | 25 |
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
Turkiye Student Evaluation Analysis - Clustering/
├── Turkiye Student Evaluation Analysis - Clustering.ipynb
├── turkiye-student-evaluation_generic.csv
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

- Open `Turkiye Student Evaluation Analysis - Clustering.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p053_*.py`:

```bash
python -m pytest tests/test_p053_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells

## 12. Cleanup Notes

Cells added during workspace standardization:

- **PyCaret Clustering** pipeline cell
