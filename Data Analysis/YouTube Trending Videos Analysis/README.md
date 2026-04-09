# YouTube Trending Videos Analysis

![Category](https://img.shields.io/badge/Category-Data%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Exploratory%20Data%20Analysis-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)

## Project Overview

**YouTube Trending Videos Analysis** is a **Exploratory Data Analysis** project in the **Data Analysis** category.

> The code suppresses warnings to avoid cluttering the output. It also configures the display options for matplotlib to enhance the quality of the displayed figures.

**Target variable:** `title_length`

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/youtube_trending/USvideos.csv` |
| Target | `title_length` |

```python
from core.data_loader import load_dataset
df = load_dataset('youtube_trending_videos_analysis')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 388 |
| `code.ipynb` | 46 code / 58 markdown cells |
| `test_youtube_trending_videos_analysis.py` | test suite |

## ML Workflow

```mermaid
graph TD
    A[Load Dataset] --> B[Exploratory Data Analysis]
    B --> C[Preprocessing]
    C --> E[Statistical Analysis]
    E --> F[Visualization]
```

## Core Logic

### Preprocessing

- Missing value imputation

### Visualizations

- Correlation heatmap
- Histograms / distributions
- Bar charts
- Scatter plots
- Word cloud

### Helper Functions

- `contains_capitalized_word()`

## Models

This project focuses on exploratory data analysis without explicit ML modeling.

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
Data Analysis/YouTube Trending Videos Analysis/
  README.md
  Youtube Trending video analysis.pdf
  code.ipynb
  data/
  guideline.txt
  pipeline.py
  test_youtube_trending_videos_analysis.py
```

## How to Run

```bash
cd "Data Analysis/YouTube Trending Videos Analysis"
python pipeline.py
```

## Testing

```bash
pytest "Data Analysis/YouTube Trending Videos Analysis/test_youtube_trending_videos_analysis.py" -v
```

## Setup

```bash
pip install matplotlib numpy pandas scikit-learn seaborn wordcloud
```

---
*README auto-generated from `code.ipynb` analysis.*