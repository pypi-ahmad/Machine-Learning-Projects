# Compound Interest Calculator

## Overview

Compound Interest Calculator is an interactive terminal tool for future value, present value, required rate, required time, loan EMI, and year-by-year growth calculations.

## Setup and run

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run python main.py
```

Enter annual rates as percentages, such as `8` for 8%. Regular contributions are treated as deposits made once per compounding period.

## Important limitation

This is an educational planning calculator. It does not account for taxes, fees, inflation, changing rates, payment timing conventions, credit risk, or personal circumstances. Do not use its output as financial advice or as the sole basis for a financial decision.

## Verification

Run `uv run python -m py_compile main.py` to check syntax. The formula helpers can be verified without using the interactive menu.
