# Twitter Sentiment Analysis - NLP

## 1. Project Overview

This project implements a **NLP / Binary Classification** pipeline for **Twitter Sentiment Analysis - NLP**.

| Property | Value |
|----------|-------|
| **ML Task** | NLP / Binary Classification |
| **Dataset Status** | BLOCKED MISSING |

## 2. Dataset

**Data sources detected in code:**

- `Twitter Sentiments.csv`

> ⚠️ **Dataset not available locally.** Twitter Sentiments.csv

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Regex text cleaning
- Text joining/concatenation
- Text stemming/lemmatization
- Text vectorization (CountVectorizer)
- Train/test split
- Data type conversion

**Models trained:**
- LogisticRegression

**Evaluation metrics:**
- Accuracy
- F1 Score

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Twitter Sentiments.csv] --> B[Regex text cleaning]
    B[Regex text cleaning] --> C[Text joining/concatenation]
    C[Text joining/concatenation] --> D[Text stemming/lemmatization]
    D[Text stemming/lemmatization] --> E[Text vectorization]
    E[Text vectorization] --> F[Train/test split]
    F[Train/test split] --> G[Data type conversion]
    G[Data type conversion] --> H[Train: LogisticRegression]
    H[Train: LogisticRegression] --> I[Evaluate: Accuracy, F1 Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 43 |
| Code cells | 36 |
| Markdown cells | 7 |
| Original models | LogisticRegression |

## 6. Model Details

### Original Models

- `LogisticRegression`

### Evaluation Metrics

- Accuracy
- F1 Score

## 7. Project Structure

```
Twitter Sentiment Analysis - NLP/
├── Twitter Sentiment Analysis - NLP.ipynb
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `nltk`
- `numpy`
- `pandas`
- `scikit-learn`
- `seaborn`
- `wordcloud`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Twitter Sentiment Analysis - NLP.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p149_*.py`:

```bash
python -m pytest tests/test_p149_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
