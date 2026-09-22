# Form Builder

Local Tkinter desktop app for creating custom forms, recording responses, and exporting selected responses to CSV.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

## Data and privacy

The app stores form definitions and responses in `forms.json` beside `main.py`. Keep that file private: its contents are not encrypted and can include information entered into forms.

Responses export only when a user selects a CSV destination. Deleting a form removes its stored responses after confirmation.

## Notes

- Python 3.13 or later is required.
- Field labels must be unique within each form.
- Required fields must be completed before submission.
