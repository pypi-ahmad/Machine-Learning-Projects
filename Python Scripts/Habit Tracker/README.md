# Habit Tracker

Local Streamlit habit tracker with daily check-ins, streaks, and 30-day summaries.

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

`habits.json` is stored beside `main.py`. Keep it private: the data is local and unencrypted. Requires Python 3.13 or later, pandas, and Streamlit.
