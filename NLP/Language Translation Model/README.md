# Language Translation Model using ML

## 1. Project Overview

This project implements a **Time Series Forecasting** pipeline for **Language Translation Model using ML**.

| Property | Value |
|----------|-------|
| **ML Task** | Time Series Forecasting |
| **Dataset Status** | BLOCKED KAGGLE |

## 2. Dataset

> ⚠️ **Dataset not available locally.** kaggle: anilreddy8989/word-guess-with-rnn

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Label mapping (function)
- Text joining/concatenation
- Word embedding layer

**Models trained:**
- Sequential

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Label mapping]
    B[Label mapping] --> C[Text joining/concatenation]
    C[Text joining/concatenation] --> D[Word embedding layer]
    D[Word embedding layer] --> E[Train: Sequential]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 21 |
| Code cells | 17 |
| Markdown cells | 4 |
| Original models | Sequential |

## 6. Model Details

### Original Models

- `Sequential`

**Neural network architecture:**

```
  Embedding
```

## 7. Project Structure

```
Language Translation Model using ML/
├── english-rnn-attempt.ipynb
├── Data.zip
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `numpy`
- `pandas`
- `tensorflow`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `english-rnn-attempt.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p147_*.py`:

```bash
python -m pytest tests/test_p147_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
- Hardcoded file paths detected — may need adjustment
- No train/test split detected in code
- No evaluation metrics found in original code
