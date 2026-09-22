"""Merge two Excel workbooks and create a Plotly pie chart.

Usage:
    uv run python script.py --show
    uv run python script.py --output purchases.html
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

PROJECT_DIR = Path(__file__).parent
DEFAULT_LEFT = PROJECT_DIR / "PriceBook.xlsx"
DEFAULT_RIGHT = PROJECT_DIR / "Purchases - Home B.xlsx"


def merged_data(left_path: Path, right_path: Path, merge_column: str) -> pd.DataFrame:
    """Load two Excel files and merge the purchase workbook with the price book."""
    left_data = pd.read_excel(left_path)
    right_data = pd.read_excel(right_path)
    if merge_column not in left_data.columns or merge_column not in right_data.columns:
        raise ValueError(f"Merge column '{merge_column}' must exist in both workbooks.")
    return right_data.merge(left_data, on=merge_column)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Excel workbooks and prepare a pie chart.")
    parser.add_argument("--left", type=Path, default=DEFAULT_LEFT, help="Price-book workbook path")
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT, help="Purchases workbook path")
    parser.add_argument("--on", default="ID", help="Shared merge-column name")
    parser.add_argument("--labels", default="MATERIAL", help="Pie-chart label column")
    parser.add_argument("--values", default="PURCHASED AMOUNT", help="Pie-chart value column")
    parser.add_argument("--show", action="store_true", help="Open the interactive chart")
    parser.add_argument("--output", type=Path, help="Write the interactive chart as HTML")
    args = parser.parse_args()

    if args.show and args.output:
        parser.error("choose either --show or --output")

    try:
        data = merged_data(args.left, args.right, args.on)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not merge workbooks: {error}") from error

    if args.labels not in data.columns or args.values not in data.columns:
        raise SystemExit("Chart label and value columns must exist in the merged data.")

    figure = px.pie(data, values=args.values, names=args.labels)
    print(f"Merged rows: {len(data)}")
    if args.output:
        figure.write_html(args.output)
        print(f"Saved chart to {args.output}")
    elif args.show:
        figure.show()
    else:
        print("Chart not opened. Use --show or --output PATH.")


if __name__ == "__main__":
    main()
