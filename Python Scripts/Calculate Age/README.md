# Calculate Age

Calculate completed age and total elapsed days from an exact birth date.

```powershell
cd "Python Scripts/Calculate Age"
uv sync
uv run python calculate.py 2000-09-22 --name Alice
```

Dates use `YYYY-MM-DD`; future dates are rejected. The project uses only the
Python standard library.
