# Employee directory

A small Streamlit app for maintaining a local CSV employee directory. It supports adding, updating, filtering, viewing, exporting, and explicitly confirmed deletion of records.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup and run

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

## Data handling

The app stores records in `employees.csv` beside `main.py`. It creates the file only after you add a record.

- Keep the folder access-controlled because contact data can be sensitive.
- Do not commit `employees.csv` if it contains real people’s information.
- Export creates a CSV download in the browser; it does not upload data anywhere.
- Deleting requires a selected record ID and a confirmation checkbox. Keep backups if the directory is important.

## Use

1. Add an employee from the sidebar. Reusing an existing employee ID updates that record.
2. Filter the directory by department, status, or text search.
3. Use the department and analytics tabs for local summaries.
4. Export the visible directory data when needed.
