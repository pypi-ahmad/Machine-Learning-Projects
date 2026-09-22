# Environment Variable Manager

A dependency-free interactive tool for viewing, searching, grouping, and managing environment variables in the current Python process.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

## What it can do

- List and search variable names.
- Group variables by prefix and inspect PATH entries.
- Set or unset a variable for the running tool process only.
- Export environment variables to `.env` or shell-script format.

## Safety behavior

- Names containing terms such as `TOKEN`, `SECRET`, `PASSWORD`, `KEY`, or `API` have their values redacted in terminal output.
- Exports exclude those sensitive-looking names by default. Exporting them requires typing `INCLUDE` at the prompt.
- Setting or unsetting a variable affects only the process running this tool. It cannot permanently change Windows user or system environment variables.
- Exports can contain private configuration. Store them outside version control and restrict access to their folder.

The key-name filter is a safety net, not a complete secret-detection system. Treat all environment values as potentially sensitive.
