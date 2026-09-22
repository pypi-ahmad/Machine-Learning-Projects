# Countdown Timer

Terminal countdown timer supporting seconds, compound units, and clock formats.

```powershell
uv sync
uv run python main.py 5m
uv run python main.py 1h30m
uv run python main.py 01:30:00
```

Run without arguments for the interactive menu. Press Ctrl+C to cancel a running timer.

Verify parsing and syntax with `uv run python -m py_compile main.py`.
