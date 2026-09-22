# Population Growth Simulator

Simulate exponential, logistic, or discrete population growth from the terminal.

## Setup

```powershell
uv sync --no-config
```

## Run

Run a bounded exponential-growth simulation:

```powershell
uv run --no-config python main.py --model exponential --p0 1000 --r 0.03 --years 50
```

Use a carrying capacity for the logistic model:

```powershell
uv run --no-config python main.py --model logistic --p0 100 --r 0.1 --K 10000 --years 100
```

The discrete model accepts separate birth and death rates:

```powershell
uv run --no-config python main.py --model discrete --p0 1000 --birth 0.08 --death 0.03 --years 50
```

Run without `--model` for interactive prompts.

## Notes

The output includes an ASCII chart, a periodic data table, and a final summary. The models are simplified educational approximations, not demographic forecasts. The tool runs locally with only the Python standard library and makes no network requests.
