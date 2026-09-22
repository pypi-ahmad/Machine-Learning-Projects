# Phonebook

A local Tkinter desktop app for storing, searching, editing, filtering, and deleting contacts.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py
```

Enter a name and phone number to save a contact. Double-click a row to edit it, use the search field and group filter to narrow the list, and select a row before deleting it.

## Data

Contacts are stored in `contacts.json` beside `main.py`. It is created after the first saved contact and is not included in the project source.

The app is local-only and makes no network requests. Treat `contacts.json` as personal data and do not commit it.
