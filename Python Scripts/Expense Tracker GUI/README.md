# Expense Tracker GUI

A local Tkinter desktop app for recording expenses, filtering the list, and viewing category summaries.

## Requirements

- Python 3.13 or later with Tkinter support
- [uv](https://docs.astral.sh/uv/)

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

## Data handling

The app stores records in `expenses.json` beside `main.py`. It creates the file only after you add an expense.

- Amounts are generic currency units; the app does not convert currencies or connect to financial institutions.
- Keep the project folder private when records contain sensitive information.
- Do not commit `expenses.json` when it contains real transaction data.
- Deleting a selected expense requires a confirmation dialog.

## Use

1. Enter a date, positive amount, description, and category.
2. Filter the table by category or month.
3. Review the total and category breakdown.
4. Select a row and confirm its deletion when needed.

The summaries are convenience views, not accounting records or financial advice.
