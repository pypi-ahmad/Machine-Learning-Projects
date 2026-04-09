# SMS Spam Detection Analysis - NLP

## 1. Project Overview

This project implements a **Classification** pipeline for **SMS Spam Detection Analysis - NLP**. The target variable is `label`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `label` |
| **Dataset Status** | BLOCKED MISSING |

## 2. Dataset

**Data sources detected in code:**

- `spam.csv`

> ⚠️ **Dataset not available locally.** spam.csv

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Stopword removal
- Lowercasing
- Regex text cleaning
- Text joining/concatenation
- Train/test split
- Text vectorization (CountVectorizer)

**Models trained:**
- LogisticRegression
- RandomForestClassifier
- SVC
- MultinomialNB

**Evaluation metrics:**
- Classification Report
- Cross-Validation Score

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load spam.csv] --> B[Stopword removal]
    B[Stopword removal] --> C[Lowercasing]
    C[Lowercasing] --> D[Regex text cleaning]
    D[Regex text cleaning] --> E[Text joining/concatenation]
    E[Text joining/concatenation] --> F[Train/test split]
    F[Train/test split] --> G[Text vectorization]
    G[Text vectorization] --> H[Train: LogisticRegression, RandomForestClassifier, SVC]
    H[Train: LogisticRegression, RandomForestClassifier, SVC] --> I[Evaluate: Classification Report, Cross-Validation Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 19 |
| Code cells | 13 |
| Markdown cells | 6 |
| Original models | LogisticRegression, RandomForestClassifier, SVC, MultinomialNB |

## 6. Model Details

### Original Models

- `LogisticRegression`
- `RandomForestClassifier`
- `SVC`
- `MultinomialNB`

### Evaluation Metrics

- Classification Report
- Cross-Validation Score

## 7. Project Structure

```
SMS Spam Detection Analysis - NLP/
├── SMS Spam Detection Analysis - NLP.ipynb
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `nltk`
- `numpy`
- `pandas`
- `scikit-learn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `SMS Spam Detection Analysis - NLP.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p090_*.py`:

```bash
python -m pytest tests/test_p090_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
