# Spreadsheet Automation

A command-line utility that merges two Excel workbooks and prepares a Plotly
pie chart from the merged data.

## Bundled workbook defaults

By default, the script reads:

- `PriceBook.xlsx`: `ID`, `Item Size`, and `Price`
- `Purchases - Home B.xlsx`: `MATERIAL`, `ID`, and `PURCHASED AMOUNT`

It performs an inner merge on `ID`, with `MATERIAL` as pie labels and
`PURCHASED AMOUNT` as pie values.

## Run

Inspect the merge without opening a browser or writing a chart:

```powershell
uv sync
uv run python script.py
```

Open the interactive chart only when wanted:

```powershell
uv run python script.py --show
```

Write a self-contained HTML chart instead:

```powershell
uv run python script.py --output purchases.html
```

## Custom workbooks

```powershell
uv run python script.py --left prices.xlsx --right purchases.xlsx --on ID --labels MATERIAL --values "PURCHASED AMOUNT" --show
```

The merge key must occur in both workbooks. The label and value columns must
occur after the merge; otherwise the script exits with a clear error.

## Dependencies

uv manages pandas, openpyxl, and Plotly in `pyproject.toml`; exact resolved
versions are recorded in `uv.lock`.
