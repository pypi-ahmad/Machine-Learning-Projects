# Inventory Manager

A local Streamlit inventory tracker for products, stock updates, reorder points, transactions, and simple analytics.

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

The app writes `inventory.csv` and `transactions.csv` beside `main.py` only after you add or update inventory. Keep backups of those files; malformed column layouts are reported instead of being silently replaced.
