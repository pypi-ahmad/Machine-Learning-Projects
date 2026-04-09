# Spreadsheet Automation

> Interactive script that merges two Excel datasets and visualizes the result as a pie chart using Plotly.

## Overview

This script prompts the user for two Excel file paths, merges them on a user-specified column, then generates an interactive pie chart based on two user-specified columns from the merged data. It uses pandas for data handling and Plotly Express for visualization.

## Features

- Reads two Excel (`.xlsx`) files via interactive user input
- Merges datasets on a user-specified common column
- Generates an interactive pie chart from the merged data
- Opens the chart automatically in the default web browser (Plotly default behavior)

## Project Structure

```
Spreadsheet_Automation/
├── script.py
├── PriceBook.xlsx
├── Purchases - Home B.xlsx
└── README.md
```

## Requirements

- Python 3.x
- `pandas` (with `openpyxl` for `.xlsx` support)
- `plotly`

## Installation

```bash
cd Spreadsheet_Automation
pip install pandas openpyxl plotly
```

## Usage

```bash
python script.py
```

The script will interactively prompt for:

1. **First dataset** — path to the first Excel file (e.g., `PriceBook.xlsx`)
2. **Second dataset** — path to the second Excel file (e.g., `Purchases - Home B.xlsx`)
3. **Basis of merging** — the common column name to merge on
4. **Criteria 1** — column for pie chart labels (names)
5. **Criteria 2** — column for pie chart values

**Example session:**

```
Enter first dataset: PriceBook.xlsx
Enter second dataset: Purchases - Home B.xlsx
What is the basis of merging? ProductID
Enter criteria 1: ProductName
Enter criteria 2: Price
```

## How It Works

1. Reads two user-provided Excel filenames via `input()`
2. Loads both files into DataFrames using `pd.read_excel()`
3. Merges the first dataset into the second on a user-specified column via `data_read_2.merge(data_read_1, on=reference)`
4. Creates a pie chart with `plotly.express.pie()` using two user-specified columns
5. Displays the chart via `fig.show()` (opens in browser)

## Configuration

No configuration files. Two sample Excel files are included:
- `PriceBook.xlsx`
- `Purchases - Home B.xlsx`

## Limitations

- All input is gathered via `input()` — no command-line arguments or config file support
- No validation of column names; a `KeyError` will occur if the user enters a non-existent column
- No error handling for missing files or unsupported file formats
- The `merge()` uses default inner join — rows without matches in both datasets are dropped silently
- Requires `openpyxl` to read `.xlsx` files, but this is not listed as an explicit dependency

## Security Notes

No security concerns identified.

## License

Not specified.
