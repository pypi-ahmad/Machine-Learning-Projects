# Telecom Churn Demo

A Streamlit learning project that trains a small logistic-regression classifier on repeatable synthetic subscriber data and displays model metrics, aggregate charts, and illustrative outputs.

## Run it

```powershell
uv sync
uv run streamlit run main.py
```

## Scope and limits

The app generates all training rows in memory. Its displayed accuracy, precision, recall, F1 score, and probability output describe only that synthetic data.

Do not use this demonstration for real customer profiling, eligibility, pricing, retention actions, or automated decisions. A real use case needs representative data, privacy controls, fairness assessment, independent validation, monitoring, and accountable human review.

## Dependencies

- Python 3.14+
- pandas
- Streamlit
