# Employee attrition demo

A Streamlit learning demo that trains a small logistic-regression model on synthetic HR-style data and displays an illustrative attrition score.

> This is not an employment decision tool. The data and model are synthetic, the score is not validated, and it must not be used to make, support, or automate decisions about people.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup and run

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

Use the profile controls to see how the demo model responds to different inputs. The other tabs show summary views of the generated training data.

## What the demo does

- Generates a repeatable synthetic dataset locally; it does not download or store employee records.
- Trains a small pure-Python logistic-regression implementation.
- Reports in-sample metrics only. They are educational diagnostics, not evidence of real-world performance.

## Limitations

- Synthetic relationships do not represent a real organization or labor market.
- The model is not evaluated on an independent test set.
- A numerical score says nothing about a person's suitability, performance, or employment outcome.
