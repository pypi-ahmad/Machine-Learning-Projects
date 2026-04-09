# Text Summarization(Medium)

## 1. Project Overview

This project implements a **NLP / Text Analysis** pipeline for **Text Summarization(Medium)**.

| Property | Value |
|----------|-------|
| **ML Task** | NLP / Text Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `tennis.csv`

**Files in project directory:**

- `tennis.csv`

**Standardized data path:** `data/text_summarizationmedium/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Word tokenization (NLTK)
- Lowercasing
- Stopword removal
- Text joining/concatenation

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load tennis.csv] --> B[Word tokenization]
    B[Word tokenization] --> C[Lowercasing]
    C[Lowercasing] --> D[Stopword removal]
    D[Stopword removal] --> E[Text joining/concatenation]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 11 |
| Code cells | 11 |
| Markdown cells | 0 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Text Summarization(Medium)/
├── Untitled.ipynb
├── tennis.csv
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

- Open `Untitled.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p037_*.py`:

```bash
python -m pytest tests/test_p037_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
- Notebook uses default name (`Untitled.ipynb`)
