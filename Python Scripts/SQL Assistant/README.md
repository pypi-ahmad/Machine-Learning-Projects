# SQL Assistant

A Streamlit SQL learning environment with a seeded, in-memory SQLite database.
It includes a query editor, schema explorer, query history, sample queries, and
pattern-based natural-language SQL hints.

## Run

```powershell
uv sync
uv run streamlit run main.py
```

The database is created in memory each time the Streamlit process starts. Its
sample customers, products, orders, order items, and employees are not saved to
disk or connected to an external database.

## Safety and scope

The editor runs statements through pandas `read_sql_query`, which is intended
for result-producing SQLite queries. Query history exists only in the current
browser session. Exported CSV files are downloaded through the browser and are
not saved by the application itself.

Natural-language hints are simple pattern matches over fixed templates. They do
not call an LLM, do not understand arbitrary requests, and should be reviewed
before they are run.

## Dependencies

uv manages pandas and Streamlit in `pyproject.toml`; exact resolved versions
are recorded in `uv.lock`.
