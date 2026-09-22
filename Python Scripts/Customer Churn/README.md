# Customer Churn Predictor

A Streamlit learning demo that trains a small logistic-regression classifier on generated customer records, then estimates churn risk from form inputs.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run streamlit run main.py
```

Use the **Predict** tab to enter tenure, monthly charge, support-call, contract, age, and product-count values. The app displays the estimated churn probability and a simple risk label.

## What the demo does

- Generates a deterministic synthetic dataset at runtime; no customer data is collected, loaded, or saved.
- Normalizes six input features and trains a pure-Python logistic-regression model.
- Shows accuracy, precision, recall, and F1 calculated on the same generated data used for training.
- Includes a sample-data tab so the generated fields and churn label are visible.

## Important limitations

The generated data and labels are illustrative, not evidence of real customer behavior. The displayed metrics are training-set metrics and are not an estimate of production performance. Do not use this app for retention, pricing, eligibility, or other business decisions without training and validating a model on appropriate real data.

## Project files

```text
main.py         # Streamlit application
pyproject.toml  # uv dependency definition
uv.lock         # Resolved dependency versions
```

## Verification

```powershell
uv run python -m py_compile main.py
uv lock --check
```
