# Survey Analyzer

A local Streamlit app for exploring survey CSV files. It summarizes responses, shows category distributions and cross-tabs, and surfaces frequent words in a selected text column.

## Run it

```powershell
uv sync
uv run streamlit run main.py
```

## Use your data

Upload a CSV from the sidebar. The app reads that file only for the current session and does not save an uploaded copy. Without an upload, it uses a deterministic in-memory sample survey; no sample CSV is written to disk.

For meaningful results, use one row per response and clear column headers. Review category labels and free-text content before interpreting charts or word frequencies.

## Dependencies

- Python 3.14+
- pandas
- Streamlit
