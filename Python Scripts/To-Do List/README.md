# To-Do List

A local terminal task manager with priorities, due dates, tags, search, and completed-task cleanup.

## Run it

```powershell
uv sync
uv run python main.py
```

Run `list` to see current tasks. Use `add <title>`, `done <id>`, `pri <id> <h|m|l>`, `due <id> <YYYY-MM-DD>`, `tag <id> <tag>`, and `find <keyword>` to manage them.

## Data

The first change creates `todos.json` beside `main.py`. The file remains local; back it up if you need your tasks elsewhere, and do not commit it if it contains private information.

## Dependencies

- Python 3.14+
- No third-party packages
