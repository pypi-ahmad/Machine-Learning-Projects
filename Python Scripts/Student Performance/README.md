# Student Performance Demo

This Streamlit project shows a small linear-regression workflow with data generated inside the app. It is a learning example, not a model for real student records, grades, eligibility, interventions, or other decisions.

## Run it

```powershell
uv sync
uv run streamlit run main.py
```

## What it does

The app generates a repeatable, synthetic dataset from study-related inputs, fits a basic regression model, and displays its illustrative score, grade band, class-level summary, and sample rows. The model includes no parent or family-background input.

The displayed R-squared and RMSE values describe only this generated dataset. They do not establish accuracy, fairness, or usefulness on real students.

## Dependencies

- Python 3.14+
- pandas
- Streamlit
