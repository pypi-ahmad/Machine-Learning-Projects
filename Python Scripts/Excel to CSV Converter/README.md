# Excel to CSV Converter

A command-line tool that converts each selected Excel worksheet into a separate CSV file.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup

```powershell
uv sync --no-config
```

## Convert a workbook

```powershell
uv run --no-config python main.py workbook.xlsx
```

By default, the converter writes one CSV for every worksheet beside the workbook. For example, `workbook.xlsx` with a `Sales` sheet becomes `workbook_Sales.csv`.

Select one worksheet or an output directory:

```powershell
uv run --no-config python main.py workbook.xlsx --sheet Sales --output converted
```

`--output` is a directory, not a CSV filename.

## Preview and overwrite behavior

Preview the planned directory without creating files:

```powershell
uv run --no-config python main.py workbook.xlsx --output converted --dry-run
```

Existing CSV files are protected. Add `--overwrite` only when you have reviewed the target files:

```powershell
uv run --no-config python main.py workbook.xlsx --output converted --overwrite
```

## List worksheets

```powershell
uv run --no-config python main.py workbook.xlsx --list-sheets
```

`.xlsx` files use `openpyxl`; legacy `.xls` files use `xlrd`. CSV inputs are reported as already converted.
