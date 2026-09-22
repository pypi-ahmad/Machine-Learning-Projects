# System Info Tool

A read-only command-line utility for viewing local operating-system, Python, disk, and network information.

## Run it

```powershell
uv sync
uv run python main.py
```

Choose an item from the interactive menu. The tool does not change system settings or files.

## Environment variables

The environment-variable view is intended for names and non-sensitive configuration. Values for variable names containing `TOKEN`, `KEY`, `SECRET`, `PASSWORD`, `PASS`, `CREDENTIAL`, or `AUTH` are redacted before display.

## Dependencies

- Python 3.14+
- No third-party packages
