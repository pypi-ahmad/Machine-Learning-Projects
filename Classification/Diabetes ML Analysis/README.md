# Machine Learning for Diabetes

## 1. Project Overview

This project implements a **Classification** pipeline for **Machine Learning for Diabetes**.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `diabetes.csv`

**Files in project directory:**

- `diabetes2.csv`

**Standardized data path:** `data/machine_learning_for_diabetes/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Train/test split
- Feature scaling (MinMaxScaler)
- Feature scaling (StandardScaler)

**Models trained:**
- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- SVC
- KNeighborsClassifier
- MLPClassifier

**Evaluation metrics:**
- Model Score

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load diabetes.csv] --> B[Train/test split]
    B[Train/test split] --> C[Feature scaling]
    C[Feature scaling] --> D[Feature scaling]
    D[Feature scaling] --> E[Train: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]
    E[Train: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier] --> F[Evaluate: Model Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 69 |
| Code cells | 33 |
| Markdown cells | 36 |
| Original models | LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, SVC, KNeighborsClassifier, MLPClassifier |

## 6. Model Details

### Original Models

- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `SVC`
- `KNeighborsClassifier`
- `MLPClassifier`

### Evaluation Metrics

- Model Score

## 7. Project Structure

```
Machine Learning for Diabetes/
├── Machine Learning for Diabetes.ipynb
├── diabetes2.csv
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

- Open `Machine Learning for Diabetes.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p145_*.py`:

```bash
python -m pytest tests/test_p145_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

No significant limitations detected.
