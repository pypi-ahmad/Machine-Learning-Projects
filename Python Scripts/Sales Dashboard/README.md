# Sales Dashboard

A Streamlit dashboard for exploring a sales CSV. It shows overall revenue and
unit metrics, monthly trends, product and regional summaries, and a filtered
CSV download.

## Run

```powershell
uv sync
uv run streamlit run main.py
```

Without an upload, the dashboard uses an in-memory sample dataset. No sample
file is written to the project directory.

## CSV columns

The dashboard looks for columns whose names include these terms:

- `date` for monthly trends
- `sales`, `amount`, or `revenue` for revenue metrics
- `unit` or `qty` for unit metrics
- `product` or `item` for product summaries
- `region` or `area` for regional summaries
- `cat` for category summaries

Columns are detected by name only. Check the imported values and totals before
using the dashboard for a business decision.

## Dependencies

Dependencies are managed with uv in `pyproject.toml` and locked in `uv.lock`.
The app requires pandas and Streamlit.
