# Bayesian Logistic Regression_bank marketing

## 1. Project Overview

This project implements a **Regression** pipeline for **Bayesian Logistic Regression_bank marketing**.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `banking.csv`

**Files in project directory:**

- `BankChurners.csv`

**Standardized data path:** `data/bayesian_logistic_regression_bank_marketing/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Data type conversion
- Text joining/concatenation

**Evaluation metrics:**
- Accuracy
- F1 Score
- Confusion Matrix

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load banking.csv] --> B[Data type conversion]
    B[Data type conversion] --> C[Text joining/concatenation]
    C[Text joining/concatenation] --> D[Evaluate: Accuracy, F1 Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 58 |
| Code cells | 43 |
| Markdown cells | 15 |

## 6. Model Details

### Evaluation Metrics

- Accuracy
- F1 Score
- Confusion Matrix

No model training in this project.

## 7. Project Structure

```
Bayesian Logistic Regression_bank marketing/
├── Bayesian Logistic Regression_bank marketing.ipynb
├── BankChurners.csv
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

- Open `Bayesian Logistic Regression_bank marketing.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p104_*.py`:

```bash
python -m pytest tests/test_p104_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
