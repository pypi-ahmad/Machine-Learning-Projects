# Tip Calculator

A terminal tool for calculating a tip, total bill, and equal per-person split. It also prints a comparison table for common tip percentages.

## Run it

```powershell
uv sync
uv run python main.py
```

Choose `1` for a single calculation or `2` for a table using 10%, 15%, 18%, 20%, and 25% tips.

## Notes

Amounts are calculated with Python floating-point values and formatted to two decimal places for display. Confirm rounding expectations before using the result to settle a payment.

## Dependencies

- Python 3.14+
- No third-party packages
