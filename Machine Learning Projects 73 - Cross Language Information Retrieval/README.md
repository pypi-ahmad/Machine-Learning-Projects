# Cross Language Information Retrieval

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **Cross Language Information Retrieval**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Standardized data path:** `data/cross_language_information_retrieval/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Word tokenization (NLTK)
- Lowercasing
- Stopword removal
- Text stemming/lemmatization
- Text joining/concatenation

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Word tokenization]
    B[Word tokenization] --> C[Lowercasing]
    C[Lowercasing] --> D[Stopword removal]
    D[Stopword removal] --> E[Text stemming/lemmatization]
    E[Text stemming/lemmatization] --> F[Text joining/concatenation]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 74 |
| Code cells | 38 |
| Markdown cells | 36 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Cross Language Information Retrieval/
├── Cross Language Information Retrieval.ipynb
├── data
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `nltk`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Cross Language Information Retrieval.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p073_*.py`:

```bash
python -m pytest tests/test_p073_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
