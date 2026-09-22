# Attendance dashboard

`main.py` is a local Streamlit dashboard for recording attendance by day and
reviewing member, monthly, and calendar summaries.

## Install and run

```powershell
cd "Python Scripts/Attendance Dashboard"
uv sync
uv run streamlit run main.py
```

Add members in the sidebar, select the attendance date, and save the checked
members. The reports and calendar tabs summarize saved presence dates. Local
data is stored in `attendance.json`, which is excluded from Git.

The app has no remote services or credentials. Automated verification uses
Streamlit's headless in-process test support rather than launching a server.
