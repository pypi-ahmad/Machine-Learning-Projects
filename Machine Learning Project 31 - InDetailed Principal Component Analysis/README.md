# InDetailed Principal Component Analysis

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **InDetailed Principal Component Analysis**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | BLOCKED LINK ONLY |

## 2. Dataset

**Data sources detected in code:**

- `load_breast_cancer() (sklearn built-in)`

> ⚠️ **Dataset not available locally.** Link-only but no downloadable URL identified

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Feature scaling (StandardScaler)
- Dimensionality reduction (PCA)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load load_breast_cancer() (sklearn built-in)] --> B[Feature scaling]
    B[Feature scaling] --> C[Dimensionality reduction]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 24 |
| Code cells | 20 |
| Markdown cells | 4 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
InDetailed Principal Component Analysis/
├── Principal Component Analysis.ipynb
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `seaborn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Principal Component Analysis.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p031_*.py`:

```bash
python -m pytest tests/test_p031_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
- No model training — this is an analysis/tutorial notebook only
