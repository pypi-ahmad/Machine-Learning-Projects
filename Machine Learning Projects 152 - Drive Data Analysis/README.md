# Drive Data Analysis

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **Drive Data Analysis**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `2018-19_AutonomousVehicleDisengagementReports(firsttimefilers).csv`
- `2019AutonomousVehicleDisengagementReports.csv`

**Files in project directory:**

- `2018-19_AutonomousVehicleDisengagementReports(firsttimefilers).csv`
- `2019AutonomousVehicleDisengagementReports.csv`
- `link_to_dataset.txt`

**Standardized data path:** `data/drive_data_analysis/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- DataFrame concatenation
- Lowercasing
- Date parsing
- Index setting

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load 2018-19_AutonomousVehicleDisengagementReports(firsttimefilers).csv] --> B[DataFrame concatenation]
    B[DataFrame concatenation] --> C[Lowercasing]
    C[Lowercasing] --> D[Date parsing]
    D[Date parsing] --> E[Index setting]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 8 |
| Code cells | 5 |
| Markdown cells | 3 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Drive Data Analysis/
├── driver-data-analysis.ipynb
├── 2018-19_AutonomousVehicleDisengagementReports(firsttimefilers).csv
├── 2019AutonomousVehicleDisengagementReports.csv
├── link_to_dataset.txt
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `seaborn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `driver-data-analysis.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p152_*.py`:

```bash
python -m pytest tests/test_p152_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
- Hardcoded file paths detected — may need adjustment
