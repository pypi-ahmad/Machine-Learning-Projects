# Expense Tracker Dashboard

A local Streamlit dashboard for recording income and expenses, filtering transactions, and viewing category and monthly summaries.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup and run

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

## Data handling

Transactions are stored in `expenses.csv` beside `main.py`. The file is created only after you add a transaction.

- Amounts use the currency units you enter. The dashboard does not apply exchange rates or currency conversion.
- Keep the project folder access-controlled if transaction records are sensitive.
- Do not commit `expenses.csv` when it contains personal or financial data.
- The dashboard does not connect to a bank, payment processor, or external service.

## Use

1. Add an income or expense transaction from the sidebar.
2. Filter transactions by type and category.
3. Review spending by category and monthly totals.
4. Review the selected transaction and tick the confirmation checkbox before deleting it.

The metrics and charts are convenience summaries, not accounting records or financial advice. Keep independent records for tax, audit, or reconciliation needs.
