# Currency Exchange Rates

An interactive CLI that scrapes an exchange-rate table from [x-rates.com](https://www.x-rates.com/). Choose a source currency and amount, then view the converted amounts listed by the site.

## Requirements

- Python 3.13+
- Internet access to x-rates.com
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run python exchange_rates.py
```

Select a currency by its displayed number, then enter a positive amount. Use a dot for decimals.

## Notes

- The script requests the live x-rates.com pages at runtime, with a 15-second timeout.
- It reports connection failures, invalid selections, invalid amounts, and a missing rate table clearly.
- The scraper depends on x-rates.com HTML structure, so site changes can require an update.
- Exchange-rate data is informational only and should not be used as financial advice or for transaction decisions without independent verification.

## Project files

```text
exchange_rates.py  # Interactive scraper entry point
pyproject.toml     # uv dependency definition
uv.lock            # Resolved dependency versions
```

## Verification

```powershell
uv run python -m py_compile exchange_rates.py
uv lock --check
```
