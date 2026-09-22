# Invoice Generator

A local Streamlit tool for creating invoice records and downloading HTML or CSV exports.

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

Invoices and the invoice counter are stored beside `main.py` in `invoices.json` and `invoice_counter.txt`. Downloaded HTML escapes entered text before rendering.
