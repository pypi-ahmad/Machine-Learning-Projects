# Contact Manager GUI

Local Tkinter contact manager with groups, favorites, search, and CSV import/export.

```powershell
uv sync
uv run python main.py
```

Contacts are stored in `contacts_manager.json` next to the app and ignored by Git. Treat CSV imports and exports as personal data.

Verify syntax with `uv run python -m py_compile main.py`.
