# Titanic Survival Prediction

## 1. Project Overview

This project implements a **Regression** pipeline for **Titanic Survival Prediction**.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Dataset Status** | OK BUILTIN |

## 2. Dataset

**Data sources detected in code:**

- `seaborn.load_dataset('titanic')`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Handle missing values (fillna)
- Label mapping (function)
- Drop columns/rows
- Outlier removal (IQR/quantile)
- Feature scaling (StandardScaler)
- Train/test split

**Models trained:**
- Sequential

**Evaluation metrics:**
- Accuracy
- Classification Report
- Confusion Matrix
- Accuracy (Keras)
- Validation loss/accuracy
- Training loss tracking

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load seaborn.load_dataset('titanic')] --> B[Handle missing values]
    B[Handle missing values] --> C[Label mapping]
    C[Label mapping] --> D[Drop columns/rows]
    D[Drop columns/rows] --> E[Outlier removal]
    E[Outlier removal] --> F[Feature scaling]
    F[Feature scaling] --> G[Train/test split]
    G[Train/test split] --> H[Train: Sequential]
    H[Train: Sequential] --> I[Evaluate: Accuracy, Classification Report]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 58 |
| Code cells | 58 |
| Markdown cells | 0 |
| Original models | Sequential |

**⚠️ Deprecated APIs detected:**

- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`

## 6. Model Details

### Original Models

- `Sequential`

**Neural network architecture:**

```
  Dropout(0.25)
```

### Evaluation Metrics

- Accuracy
- Classification Report
- Confusion Matrix
- Accuracy (Keras)
- Validation loss/accuracy
- Training loss tracking

## 7. Project Structure

```
Titanic Survival Prediction/
├── Untitled.ipynb
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

Automated tests are available in `tests/test_p040_*.py`:

```bash
python -m pytest tests/test_p040_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- `sns.distplot()` is deprecated — use `sns.displot()` or `sns.histplot()`
- Notebook uses default name (`Untitled.ipynb`)
