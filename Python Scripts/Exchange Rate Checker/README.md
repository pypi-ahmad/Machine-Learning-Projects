# Exchange Rate Checker

A dependency-free command-line tool for looking up reference exchange rates and converting amounts with ExchangeRate-API's open endpoint.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup

```powershell
uv sync --no-config
```

## Convert an amount

```powershell
uv run --no-config python main.py --from USD --to EUR --amount 100
```

Show rates for a base currency:

```powershell
uv run --no-config python main.py --base EUR
```

Run without arguments for interactive commands:

```powershell
uv run --no-config python main.py
```

## Provider and limitations

The app calls the open endpoint documented by [ExchangeRate-API](https://www.exchangerate-api.com/docs/free). The provider requires attribution, which the tool prints with each result. Its open data updates once per day and is rate-limited, so cache results and avoid repeated polling.

The displayed rate is a reference conversion only. It does not include bank spreads, fees, taxes, settlement timing, or a guaranteed executable price. Do not use it as the sole basis for a financial transaction.
