# Captcha Recognition

## 1. Project Overview

This project implements a **Image Classification** pipeline for **Captcha Recognition**.

| Property | Value |
|----------|-------|
| **ML Task** | Image Classification |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Files in project directory:**

- `linkt_to_dataset.txt`

**Standardized data path:** `data/captcha_recognition/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Evaluation metrics:**
- Accuracy (Keras)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Evaluate: Accuracy (Keras)]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 13 |
| Code cells | 13 |
| Markdown cells | 0 |

## 6. Model Details

### Evaluation Metrics

- Accuracy (Keras)

No model training in this project.

## 7. Project Structure

```
Captcha Recognition/
├── captcha-recognition.ipynb
├── linkt_to_dataset.txt
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `keras`
- `matplotlib`
- `numpy`
- `opencv-python`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `captcha-recognition.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p155_*.py`:

```bash
python -m pytest tests/test_p155_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
