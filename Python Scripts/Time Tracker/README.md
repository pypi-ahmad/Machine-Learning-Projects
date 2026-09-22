# Time Tracker

A local Streamlit app for logging project time, viewing daily and weekly summaries, and exporting a timesheet CSV.

## Run it

```powershell
uv sync
uv run streamlit run main.py
```

## Data

Entries are saved to `time_log.csv` beside `main.py` after you log the first entry. The app does not send the log anywhere. Back up that file if you need to retain local records, and do not commit it if it contains private work details.

## Features

- Add dated time entries with start and end times
- Filter the log by project and date range
- View project, daily, and weekly summaries
- Download the current log as a CSV timesheet

## Dependencies

- Python 3.14+
- pandas
- Streamlit
