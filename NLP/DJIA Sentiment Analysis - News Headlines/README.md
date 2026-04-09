# Sentiment Analysis - Dow Jones (DJIA) Stock using News Headlines

## 1. Project Overview

This project implements a **Classification** pipeline for **Sentiment Analysis - Dow Jones (DJIA) Stock using News Headlines**. The target variable is `Label`.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Target Variable** | `Label` |
| **Dataset Status** | OK LOCAL |
| **Standardized Pipeline** | Yes (LazyPredict + PyCaret) |

## 2. Dataset

**Data sources detected in code:**

- `Stock Headlines.csv`

**Files in project directory:**

- `Stock Headlines.csv`

**Standardized data path:** `data/sentiment_analysis_-_dow_jones_djia_stock_using_news_headlines/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop missing values (dropna)
- Index setting
- Date-based train/test split
- Regex text cleaning (pandas)
- Lowercasing
- Text joining/concatenation
- Stopword removal
- Text stemming/lemmatization
- Text vectorization (CountVectorizer)

### Standardized Pipeline (added)

- **LazyClassifier**: Automated comparison of multiple models in a single call
- **PyCaret Classification**: Full AutoML pipeline (setup → compare → tune → evaluate → finalize)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Stock Headlines.csv] --> B[Drop missing values]
    B[Drop missing values] --> C[Index setting]
    C[Index setting] --> D[Date-based train/test split]
    D[Date-based train/test split] --> E[Regex text cleaning]
    E[Regex text cleaning] --> F[Lowercasing]
    F[Lowercasing] --> G[Text joining/concatenation]
    G[Text joining/concatenation] --> H[Stopword removal]
    H[Stopword removal] --> I[Text stemming/lemmatization]
    I[Text stemming/lemmatization] --> J[Text vectorization]
    J[Text vectorization] --> K[LazyClassifier Benchmark]
    K[LazyClassifier Benchmark] --> L[PyCaret Classification]
    L[PyCaret Classification] --> M[Evaluate]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 51 |
| Code cells | 40 |
| Markdown cells | 11 |
| Original cells | 38 |
| Standardized cells (added) | 13 |
| Original model training | None — preprocessing/EDA only |

## 6. Model Details

### LazyClassifier (Standardized)

Compares 20+ classifiers, ranked by accuracy/F1.

### PyCaret Classification (Standardized)

AutoML pipeline: `setup()` → `compare_models()` → `tune_model()` → `finalize_model()`

> ⚠️ Requires Python ≤ 3.11.

## 7. Project Structure

```
Sentiment Analysis - Dow Jones (DJIA) Stock using News Headlines/
├── Stock Sentiment Analysis.ipynb
├── Stock Headlines.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `lazypredict`
- `matplotlib`
- `nltk`
- `numpy`
- `pandas`
- `pycaret`
- `scikit-learn`
- `seaborn`
- `wordcloud`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Stock Sentiment Analysis.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p084_*.py`:

```bash
python -m pytest tests/test_p084_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- PyCaret cells require Python ≤ 3.11 — they will fail on Python 3.12+
- No original model training exists — only auto-generated LazyPredict/PyCaret cells
- Hardcoded file paths detected — may need adjustment
- Contains Google Colab artifacts

## 12. Cleanup Notes

Cells added during workspace standardization:

- **LazyClassifier** benchmark cell
- **PyCaret Classification** pipeline cell
