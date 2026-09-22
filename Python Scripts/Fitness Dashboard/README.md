# Fitness Dashboard

A local Streamlit dashboard for recording workouts, weight entries, and personal activity goals.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup and run

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

## Data handling

Records are stored in `fitness.json` beside `main.py`. The file is created only after you save a workout, weight entry, or goal.

- Keep the project folder private if fitness or weight data is sensitive.
- Do not commit `fitness.json` when it contains real personal records.
- The dashboard does not connect to wearables, health providers, or external services.
- Workout estimates, calories, streaks, and goals are personal tracking aids, not medical, nutrition, or training advice.

## Use

1. Log workouts with date, duration, optional calories and distance, and notes.
2. Add or update a weight entry for a date.
3. Review local activity summaries and charts.
4. Set weekly workout, time, and optional target-weight goals.
