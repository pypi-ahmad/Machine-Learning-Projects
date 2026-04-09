# 4 Mall customer segmentation

![Category](https://img.shields.io/badge/Category-Clustering-blue)
![Task](https://img.shields.io/badge/Task-Clustering-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**4 Mall customer segmentation** is a **Clustering** project in the **Clustering** category.

> Automated clustering pipeline with PyCaret:

**Models:** KMeans, PyCaret

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/mall_customer_segmentation/Mall_Customers.csv` |

```python
from core.data_loader import load_dataset
df = load_dataset('mall_customer_segmentation')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 191 |
| `train.py` | 167 |
| `evaluate.py` | 167 |
| `4 Mall customer segmentation.ipynb` | 18 code / 10 markdown cells |
| `test_mall_customer_segmentation.py` | test suite |

## ML Workflow

```mermaid
graph TD
    A[Load Dataset] --> B[Exploratory Data Analysis]
    B --> C[Preprocessing]
    C --> E[Clustering]
    E --> F[AutoML Comparison]
    F --> G[Cluster Evaluation]
    G --> H[Visualization]
```

## Core Logic

### Preprocessing

- Missing value imputation

### Visualizations

- Histograms / distributions
- Count plots
- Box plots
- Scatter plots
- Elbow method
- Silhouette analysis

## Models

| Model | Type |
|-------|------|
| KMeans | Centroid Clustering |
| PyCaret | AutoML Framework |

AutoML is toggled via the `USE_AUTOML` flag in pipeline scripts.
**PyCaret** `compare_models()` runs cross-validated comparison.

## Reproducibility

```python
random.seed(42); np.random.seed(42); os.environ['PYTHONHASHSEED'] = '42'
```

```bash
python pipeline.py --seed 123    # custom seed
python pipeline.py --reproduce   # locked seed=42
```

## Project Structure

```
Clustering/4 Mall customer segmentation/
  4 Mall customer segmentation.docx
  4 Mall customer segmentation.ipynb
  Mall customer segmentation.pdf
  README.md
  evaluate.py
  pipeline.py
  test_mall_customer_segmentation.py
  train.py
```

## How to Run

```bash
cd "Clustering/4 Mall customer segmentation"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Clustering/4 Mall customer segmentation/test_mall_customer_segmentation.py" -v
```

## Setup

```bash
pip install matplotlib numpy pandas pycaret scikit-learn seaborn
```

---
*README auto-generated from `4 Mall customer segmentation.ipynb` analysis.*