# Personal Finance Dashboard

A local Streamlit dashboard for recording income and expenses, reviewing spending, tracking monthly trends, and exporting transactions.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config streamlit run main.py
```

Use the sidebar form to add an income or expense. The dashboard then shows balance and savings metrics, transactions, category spending, monthly trends, and a CSV download.

## Data

Transactions are stored in `finance.csv` beside `main.py`. The file is created after the first saved transaction and is not included in the project source.

The dashboard is local-only and makes no network requests. Treat `finance.csv` as personal financial data and do not commit it.
