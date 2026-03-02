# Imdb sentiment review Analysis using ML

## 1. Project Overview

This project implements a **NLP / Binary Classification** pipeline for **Imdb sentiment review Analysis using ML**.

| Property | Value |
|----------|-------|
| **ML Task** | NLP / Binary Classification |
| **Dataset Status** | BLOCKED LINK ONLY |

## 2. Dataset

> ⚠️ **Dataset not available locally.** Link-only but no downloadable URL identified

## 3. Pipeline Overview

### Original Notebook Pipeline

**Models trained:**
- Sequential

**Evaluation metrics:**
- Accuracy (Keras)
- Training loss tracking

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Train: Sequential]
    B[Train: Sequential] --> C[Evaluate: Accuracy (Keras), Training loss tracking]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 12 |
| Code cells | 12 |
| Markdown cells | 0 |
| Original models | Sequential |

## 6. Model Details

### Original Models

- `Sequential`

**Neural network architecture:**

```
  Dense(16)
  Dense(1)
```

### Evaluation Metrics

- Accuracy (Keras)
- Training loss tracking

## 7. Project Structure

```
Imdb sentiment review Analysis using ML/
├── Untitled.ipynb
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `tensorflow`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Untitled.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p006_*.py`:

```bash
python -m pytest tests/test_p006_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
- Notebook uses default name (`Untitled.ipynb`)
- No train/test split detected in code
