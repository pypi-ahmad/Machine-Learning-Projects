# Subscription Tracker

A local Streamlit app for recording recurring subscriptions, estimating monthly and yearly cost, and listing renewal dates.

## Run it

```powershell
uv sync
uv run streamlit run main.py
```

## Data

The app stores subscriptions in `subscriptions.json` next to `main.py`. The file is created only after the first subscription is saved and is not included in version control. Keep a copy of it if you want to retain your local records.

## Dependencies

- Python 3.14+
- pandas
- Streamlit
