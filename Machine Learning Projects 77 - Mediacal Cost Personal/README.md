# Mediacal Cost Personal

## 1. Project Overview

This project implements a **Regression** pipeline for **Mediacal Cost Personal**.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `insurance.csv`

**Files in project directory:**

- `insurance.csv`
- `link_to_dataset.txt`

**Standardized data path:** `data/mediacal_cost_personal/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Label mapping (function)
- Train/test split
- Feature scaling (StandardScaler)

**Models trained:**
- LinearRegression

**Evaluation metrics:**
- R² Score

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load insurance.csv] --> B[Label mapping]
    B[Label mapping] --> C[Train/test split]
    C[Train/test split] --> D[Feature scaling]
    D[Feature scaling] --> E[Train: LinearRegression]
    E[Train: LinearRegression] --> F[Evaluate: R² Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 28 |
| Code cells | 28 |
| Markdown cells | 0 |
| Original models | LinearRegression |

## 6. Model Details

### Original Models

- `LinearRegression`

### Evaluation Metrics

- R² Score

## 7. Project Structure

```
Mediacal Cost Personal/
├── medical-cost-personal-datasets(1).ipynb
├── insurance.csv
├── link_to_dataset.txt
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

- Open `medical-cost-personal-datasets(1).ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p077_*.py`:

```bash
python -m pytest tests/test_p077_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

No significant limitations detected.
