# Goal Tracker

Local Streamlit app for tracking goals, milestones, deadlines, and progress.

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

Goals are stored as `goals.json` beside `main.py`. Keep personal goal data private; it is not encrypted or synchronized to an external service. Requires Python 3.13 or later, pandas, and Streamlit.
