# SQL Formatter

An offline command-line tool for normalizing SQL whitespace, formatting common
keywords, checking a small set of risky patterns, and extracting basic query
facts. It never connects to or executes against a database.

## Run

Format inline SQL:

```powershell
uv run python main.py --inline "select id, name from customers where id = 1"
```

Lint a SQL file:

```powershell
uv run python main.py --file query.sql --lint
```

Write formatted SQL to a separate file:

```powershell
uv run python main.py --file query.sql --output formatted.sql
```

Run `uv run python main.py` without arguments for the interactive prompt.

## Scope and limitations

Formatting and linting are regular-expression based, not a full SQL parser.
The tool can misinterpret dialect-specific syntax, nested constructs, quoted
identifiers, comments, or SQL keywords inside string literals. Review the
result before running it against a database.

Warnings flag patterns such as `SELECT *`, `DROP`, and `DELETE` or `UPDATE`
without `WHERE`; they are not complete security or correctness checks.

## Dependencies

The project uses only Python's standard library. uv records the Python
requirement and provides the reproducible environment.
