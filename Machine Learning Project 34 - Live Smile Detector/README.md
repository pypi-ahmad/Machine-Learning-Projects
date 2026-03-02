# Live Smile Detector

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **Live Smile Detector**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | BLOCKED MISSING |

## 2. Dataset

> ⚠️ **Dataset not available locally.** frontal_face.xml (OpenCV cascade, can be resolved from opencv data)

## 3. Pipeline Overview

The original notebook primarily contains data loading and exploratory data analysis.

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Explore / Visualize]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 0 |
| Code cells | 0 |
| Markdown cells | 0 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Live Smile Detector/
├── Smile.py
├── smile.xml
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `opencv-python`

## 9. How to Run

Run the Python script(s):

```bash
python "Smile.py"
```

## 10. Testing

Automated tests are available in `tests/test_p034_*.py`:

```bash
python -m pytest tests/test_p034_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
- No model training — this is an analysis/tutorial notebook only
- Hardcoded file paths detected — may need adjustment
