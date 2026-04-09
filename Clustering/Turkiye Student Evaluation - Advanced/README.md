# Clustering -Turkiye Student Evaluation Analysis - Clustering

## 1. Project Overview

This project implements a **Clustering** pipeline for **Clustering -Turkiye Student Evaluation Analysis - Clustering**.

| Property | Value |
|----------|-------|
| **ML Task** | Clustering |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `turkiye-student-evaluation_generic.csv`

**Files in project directory:**

- `turkiye-student-evaluation_generic.csv`

**Standardized data path:** `data/clustering_-turkiye_student_evaluation_analysis_-_clustering/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Index setting
- Dimensionality reduction (PCA)

**Models trained:**
- KMeans
- AgglomerativeClustering

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load turkiye-student-evaluation_generic.csv] --> B[Index setting]
    B[Index setting] --> C[Dimensionality reduction]
    C[Dimensionality reduction] --> D[Train: KMeans, AgglomerativeClustering]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 36 |
| Code cells | 28 |
| Markdown cells | 8 |
| Original models | KMeans, AgglomerativeClustering |

## 6. Model Details

### Original Models

- `KMeans`
- `AgglomerativeClustering`

## 7. Project Structure

```
Clustering/
├── Turkiye Student Evaluation Analysis - Clustering.ipynb
├── turkiye-student-evaluation_generic.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `seaborn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Turkiye Student Evaluation Analysis - Clustering.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p089_*.py`:

```bash
python -m pytest tests/test_p089_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- No evaluation metrics found in original code
