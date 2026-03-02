# Seattle Hotels Recommender

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **Seattle Hotels Recommender**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `Seattle_Hotels.csv`

**Files in project directory:**

- `Seattle_Hotels.csv`

**Standardized data path:** `data/seattle_hotels_recommender/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Text vectorization (CountVectorizer)
- Stopword removal
- Lowercasing
- Text joining/concatenation
- Index setting
- TF-IDF vectorization

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Seattle_Hotels.csv] --> B[Text vectorization]
    B[Text vectorization] --> C[Stopword removal]
    C[Stopword removal] --> D[Lowercasing]
    D[Lowercasing] --> E[Text joining/concatenation]
    E[Text joining/concatenation] --> F[Index setting]
    F[Index setting] --> G[TF-IDF vectorization]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 45 |
| Code cells | 27 |
| Markdown cells | 18 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Seattle Hotels Recommender/
├── Seattle Hotels Recommender.ipynb
├── Seattle_Hotels.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `nltk`
- `numpy`
- `pandas`
- `plotly`
- `scikit-learn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Seattle Hotels Recommender.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p119_*.py`:

```bash
python -m pytest tests/test_p119_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
