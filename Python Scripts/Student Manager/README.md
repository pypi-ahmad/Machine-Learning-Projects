# Student Manager

A local Tkinter application for recording basic student details, subject grades,
calculated GPA values, and summary statistics.

## Run

```powershell
uv run python main.py
```

Use the student list to search, edit, or delete records. The add/edit form
collects basic fields and one letter grade for each built-in subject.

## Data and calculations

Records are stored in `students.json` beside `main.py`. The app computes an
unweighted GPA from its built-in grade-point mapping and shows fixed-threshold
summary categories such as honor roll and at-risk.

These values are local heuristics, not an official transcript, institutional
grade policy, academic-risk decision, or student-information system.

## Privacy

The local JSON file can contain names, email addresses, ages, grades, and
academic notes. It is plaintext, not access-controlled, and not synchronized.
Use only with appropriate permission and avoid storing sensitive records on a
shared device.

## Dependencies

The project uses only Python's standard library, including Tkinter. uv records
the Python requirement and provides the reproducible environment.
