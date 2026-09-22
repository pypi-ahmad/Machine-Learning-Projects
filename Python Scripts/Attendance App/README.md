# Attendance App

`main.py` is a local Tkinter attendance tracker for students or employees. Add
members, mark daily attendance, and generate a report for a chosen date range.

## Install and run

```powershell
cd "Python Scripts/Attendance App"
uv sync
uv run python main.py
```

The app stores its local data in `attendance.json`, which is excluded from Git.

## Workflow

1. Add members on the **Members** tab.
2. Select a date on **Mark Attendance**, set each status, and save it.
3. Choose inclusive start and end dates on **Reports** to generate counts and
   the percentage of recorded days marked Present.

Dates must use `YYYY-MM-DD`. The report includes only dates for which a member
has a saved status; it does not infer absences for missing days.

The project uses only the Python standard library. The interface is not started
by automated verification.
