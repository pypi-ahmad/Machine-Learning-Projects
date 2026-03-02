# Earthquake Prediction

## 1. Project Overview

This project implements a **Regression** pipeline for **Earthquake Prediction**.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `database.csv`

**Files in project directory:**

- `database.csv`

**Standardized data path:** `data/earthquake_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Date parsing
- Drop columns/rows
- Drop missing values (dropna)
- Train/test split

**Models trained:**
- Sequential

**Evaluation metrics:**
- Accuracy (Keras)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load database.csv] --> B[Date parsing]
    B[Date parsing] --> C[Drop columns/rows]
    C[Drop columns/rows] --> D[Drop missing values]
    D[Drop missing values] --> E[Train/test split]
    E[Train/test split] --> F[Train: Sequential]
    F[Train: Sequential] --> G[Evaluate: Accuracy (Keras)]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 23 |
| Code cells | 23 |
| Markdown cells | 0 |
| Original models | Sequential |

## 6. Model Details

### Original Models

- `Sequential`

**Neural network architecture:**

```
  Dense(128)
  Dense(64)
  Dense(32)
  Dense(16)
  Dense(2)
```

### Evaluation Metrics

- Accuracy (Keras)

## 7. Project Structure

```
Earthquake Prediction/
├── Untitled.ipynb
├── database.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `keras`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `seaborn`
- `tensorflow`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Untitled.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p023_*.py`:

```bash
python -m pytest tests/test_p023_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Notebook uses default name (`Untitled.ipynb`)
