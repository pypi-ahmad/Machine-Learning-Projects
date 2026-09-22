# Student Grade Tracker

A local Streamlit app for recording student scores by subject, applying the
built-in grade scale, viewing summaries, and downloading a CSV export.

## Run

```powershell
uv sync
uv run streamlit run main.py
```

Add a student name, subject, score, date, and optional note from the sidebar.
Scores must be between 0 and 100 through the app's slider.

## Data and calculations

Entries are stored in `grades.csv` beside `main.py` after the first saved grade.
The app derives a letter grade and GPA value from its included scale, then
reports arithmetic averages by student or subject.

The GPA is an unweighted average: the app has no course-credit or institutional
grading-policy model. It is a personal tracking aid, not an official academic
record, transcript, or grading system.

## Privacy

Student names, scores, dates, and notes are plaintext local data. They are not
encrypted, access-controlled, or synchronized. Use only with appropriate
consent and avoid storing sensitive student information.

## Dependencies

uv manages pandas and Streamlit in `pyproject.toml`; exact resolved versions
are recorded in `uv.lock`.
