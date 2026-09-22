# Desktop Notification Application

A small CLI reminder that sends a desktop notification through [plyer](https://plyer.readthedocs.io/). It repeats hourly by default until you stop it.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- A desktop environment that supports notifications

## Run

From this directory:

```powershell
uv sync
uv run python "Desktop Notification Application.py"
```

Stop the repeating reminder with `Ctrl+C`.

## Options

```powershell
# Send one reminder, then exit
uv run python "Desktop Notification Application.py" --once

# Send a custom reminder every 30 minutes
uv run python "Desktop Notification Application.py" --interval 1800 --title "Stretch" --message "Time to stand up."
```

`--interval` and `--timeout` must be positive whole seconds.

## Project files

```text
Desktop Notification Application.py  # CLI entry point
pyproject.toml                       # uv dependency definition
uv.lock                              # Resolved dependency versions
```

## Verification

```powershell
uv run python "Desktop Notification Application.py" --help
uv run python -m py_compile "Desktop Notification Application.py"
uv lock --check
```
