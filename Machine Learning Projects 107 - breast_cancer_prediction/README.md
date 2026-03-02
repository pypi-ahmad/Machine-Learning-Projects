# breast_cancer_prediction

## 1. Project Overview

This project implements a **Regression** pipeline for **breast_cancer_prediction**.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `breast_cancer.csv`

**Files in project directory:**

- `Wisconsin-bc-data.csv`

**Standardized data path:** `data/breast_cancer_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Train/test split
- Feature scaling (MinMaxScaler)

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
    A[Load breast_cancer.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Train/test split]
    C[Train/test split] --> D[Feature scaling]
    D[Feature scaling] --> E[Train: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]
    E[Train: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier] --> F[Evaluate: Model Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 65 |
| Code cells | 33 |
| Markdown cells | 32 |
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
breast_cancer_prediction/
├── breast_cancer_predict.ipynb
├── Wisconsin-bc-data.csv
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

- Open `breast_cancer_predict.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p107_*.py`:

```bash
python -m pytest tests/test_p107_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

No significant limitations detected.
