# Probability Simulator

Run Monte Carlo simulations for coin flips, dice rolls, card draws, the birthday problem, and the Monty Hall problem.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py --coin 1000
uv run --no-config python main.py --dice 2 10000
uv run --no-config python main.py --birthday 23 10000
uv run --no-config python main.py --monty 10000
uv run --no-config python main.py --cards 5 10000
```

Run without arguments for interactive prompts.

## Notes

Results vary because the simulations use random sampling. Larger trial counts usually produce estimates closer to the theoretical probabilities, but take longer to run.

The tool uses only the Python standard library, runs locally, and makes no network requests.
