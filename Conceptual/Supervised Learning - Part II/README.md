# (Conceptual) Supervised Learning - Part II

## 1. Project Overview

This project implements a **Classification** pipeline for **(Conceptual) Supervised Learning - Part II**.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `fruit_data_with_colors.txt`
- `load_breast_cancer() (sklearn built-in)`
- `load_breast_cancer() (sklearn built-in)`

**Files in project directory:**

- `fruit_data_with_colors.txt`

**Standardized data path:** `data/conceptual_supervised_learning_-_part_ii/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Train/test split
- Feature scaling (MinMaxScaler)

**Models trained:**
- RandomForestClassifier
- GradientBoostingClassifier
- GaussianNB
- MLPClassifier
- MLPRegressor

**Evaluation metrics:**
- Model Score

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load fruit_data_with_colors.txt] --> B[Train/test split]
    B[Train/test split] --> C[Feature scaling]
    C[Feature scaling] --> D[Train: RandomForestClassifier, GradientBoostingClassifier, GaussianNB]
    D[Train: RandomForestClassifier, GradientBoostingClassifier, GaussianNB] --> E[Evaluate: Model Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 18 |
| Code cells | 18 |
| Markdown cells | 0 |
| Original models | RandomForestClassifier, GradientBoostingClassifier, GaussianNB, MLPClassifier, MLPRegressor |

## 6. Model Details

### Original Models

- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `GaussianNB`
- `MLPClassifier`
- `MLPRegressor`

### Evaluation Metrics

- Model Score

## 7. Project Structure

```
(Conceptual) Supervised Learning - Part II/
├── Supervised Learning - Part II.ipynb
├── fruit_data_with_colors.txt
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

- Open `Supervised Learning - Part II.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p123_*.py`:

```bash
python -m pytest tests/test_p123_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

No significant limitations detected.
