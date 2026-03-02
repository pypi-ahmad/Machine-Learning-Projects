# NLP Disaster or Not Disaster

## 1. Project Overview

This project implements a **NLP / Binary Classification** pipeline for **NLP Disaster or Not Disaster**.

| Property | Value |
|----------|-------|
| **ML Task** | NLP / Binary Classification |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Files in project directory:**

- `train.csv`

**Standardized data path:** `data/nlp_disaster_or_not_disaster/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Label mapping (function)
- Stopword removal
- Lowercasing
- Text joining/concatenation
- Text tokenization (Keras)
- Sequence padding
- Word embedding layer

**Models trained:**
- Sequential

**Evaluation metrics:**
- Accuracy (Keras)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Label mapping]
    B[Label mapping] --> C[Stopword removal]
    C[Stopword removal] --> D[Lowercasing]
    D[Lowercasing] --> E[Text joining/concatenation]
    E[Text joining/concatenation] --> F[Text tokenization]
    F[Text tokenization] --> G[Sequence padding]
    G[Sequence padding] --> H[Word embedding layer]
    H[Word embedding layer] --> I[Train: Sequential]
    I[Train: Sequential] --> J[Evaluate: Accuracy (Keras)]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 36 |
| Code cells | 36 |
| Markdown cells | 0 |
| Original models | Sequential |

## 6. Model Details

### Original Models

- `Sequential`

**Neural network architecture:**

```
  LSTM(64)
  Dense(1)
  Embedding
```

### Evaluation Metrics

- Accuracy (Keras)

## 7. Project Structure

```
Machine Learning Project 27- NLP Disaster or Not Disaster/
├── NLP.ipynb
├── train.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `keras`
- `matplotlib`
- `nltk`
- `numpy`
- `pandas`
- `tensorflow`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `NLP.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p027_*.py`:

```bash
python -m pytest tests/test_p027_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Hardcoded file paths detected — may need adjustment
