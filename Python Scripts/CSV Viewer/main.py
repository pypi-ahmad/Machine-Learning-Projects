"""CSV Viewer — CLI tool.

Load, display, filter, sort, summarize and export CSV files
without any external dependencies.

Usage:
    python main.py
    python main.py data.csv
"""

import csv
import io
import statistics
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_csv(path: Path, delimiter: str = ",") -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows)."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def save_csv(path: Path, headers: list[str], rows: list[list[str]],
             delimiter: str = ",") -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(headers)
        writer.writerows(rows)


def col_index(headers: list[str], name: str) -> int:
    name_l = name.strip().lower()
    for i, h in enumerate(headers):
        if h.strip().lower() == name_l:
            return i
    try:
        return int(name)
    except ValueError:
        return -1


def filter_rows(rows: list[list[str]], col: int,
                value: str, mode: str = "contains") -> list[list[str]]:
    v = value.lower()
    if mode == "eq":
        return [r for r in rows if len(r) > col and r[col].strip().lower() == v]
    elif mode == "neq":
        return [r for r in rows if len(r) > col and r[col].strip().lower() != v]
    elif mode == "gt":
        try:
            return [r for r in rows if len(r) > col and float(r[col]) > float(value)]
        except ValueError:
            return rows
    elif mode == "lt":
        try:
            return [r for r in rows if len(r) > col and float(r[col]) < float(value)]
        except ValueError:
            return rows
    return [r for r in rows if len(r) > col and v in r[col].strip().lower()]


def sort_rows(rows: list[list[str]], col: int,
              reverse: bool = False, numeric: bool = False) -> list[list[str]]:
    def key(r):
        val = r[col] if len(r) > col else ""
        if numeric:
            try:
                return float(val)
            except ValueError:
                return 0.0
        return val.lower()
    return sorted(rows, key=key, reverse=reverse)


def summarize_col(rows: list[list[str]], col: int, header: str) -> None:
    vals_raw = [r[col] for r in rows if len(r) > col and r[col].strip()]
    print(f"\n  Column: {header}")
    print(f"  Count : {len(vals_raw)}")
    nums = []
    for v in vals_raw:
        try:
            nums.append(float(v))
        except ValueError:
            pass
    if nums:
        print(f"  Min   : {min(nums):.4g}")
        print(f"  Max   : {max(nums):.4g}")
        print(f"  Mean  : {statistics.mean(nums):.4g}")
        print(f"  Median: {statistics.median(nums):.4g}")
        if len(nums) > 1:
            print(f"  Stdev : {statistics.stdev(nums):.4g}")
    else:
        # Categorical
        from collections import Counter
        freq = Counter(vals_raw).most_common(5)
        print(f"  Top values:")
        for val, cnt in freq:
            print(f"    '{val}': {cnt}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_table(headers: list[str], rows: list[list[str]],
                max_rows: int = 25, max_col: int = 20) -> None:
    if not headers:
        print("  (empty)")
        return
    widths = [min(max_col, max(len(h), max(
        (len(r[i]) if i < len(r) else 0) for r in rows[:max_rows]
    ) if rows else len(h))) for i, h in enumerate(headers)]
    sep = "  " + "-+-".join("-" * w for w in widths)
    header_row = "  " + " | ".join(h[:widths[i]].ljust(widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print(sep)
    for row in rows[:max_rows]:
        cells = []
        for i, w in enumerate(widths):
            v = row[i] if i < len(row) else ""
            cells.append(v[:w].ljust(w))
        print("  " + " | ".join(cells))
    if len(rows) > max_rows:
        print(f"\n  ... {len(rows) - max_rows} more rows (showing {max_rows} of {len(rows)})")
    print(f"\n  {len(rows)} row(s), {len(headers)} column(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
CSV Viewer
----------
1. View data
2. Filter rows
3. Sort rows
4. Column summary
5. Save filtered/sorted data
6. Load different file
0. Quit
"""


def main() -> None:
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    headers: list[str] = []
    rows: list[list[str]] = []
    current_path: Path | None = None

    def load(path: Path) -> bool:
        nonlocal headers, rows, current_path
        if not path.exists():
            print(f"  File not found: {path}")
            return False
        delim = "," if path.suffix.lower() != ".tsv" else "\t"
        headers, rows = load_csv(path, delim)
        current_path = path
        print(f"  Loaded {len(rows)} rows, {len(headers)} columns from {path.name}")
        return True

    if path_arg:
        load(Path(path_arg))
    else:
        path_str = input("  CSV file path: ").strip().strip('"')
        if path_str:
            load(Path(path_str))

    print("CSV Viewer")
    work_rows = list(rows)

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            if not headers:
                print("  No file loaded.")
                continue
            n_str = input("  Rows to show (default 25): ").strip()
            n = int(n_str) if n_str.isdigit() else 25
            print_table(headers, work_rows, n)

        elif choice == "2":
            if not headers:
                print("  No file loaded.")
                continue
            print(f"  Columns: {', '.join(f'{i}:{h}' for i, h in enumerate(headers))}")
            col_name = input("  Column name or index: ").strip()
            ci = col_index(headers, col_name)
            if ci < 0:
                print("  Column not found.")
                continue
            print("  Modes: contains / eq / neq / gt / lt")
            mode = input("  Mode (default contains): ").strip().lower() or "contains"
            value = input("  Value: ")
            work_rows = filter_rows(work_rows, ci, value, mode)
            print(f"  {len(work_rows)} row(s) after filter.")

        elif choice == "3":
            if not headers:
                print("  No file loaded.")
                continue
            print(f"  Columns: {', '.join(f'{i}:{h}' for i, h in enumerate(headers))}")
            col_name = input("  Sort by column: ").strip()
            ci = col_index(headers, col_name)
            if ci < 0:
                print("  Column not found.")
                continue
            rev  = input("  Descending? (y/n, default n): ").strip().lower() == "y"
            num  = input("  Numeric sort? (y/n, default n): ").strip().lower() == "y"
            work_rows = sort_rows(work_rows, ci, rev, num)
            print(f"  Sorted {len(work_rows)} rows.")

        elif choice == "4":
            if not headers:
                print("  No file loaded.")
                continue
            print(f"  Columns: {', '.join(f'{i}:{h}' for i, h in enumerate(headers))}")
            col_name = input("  Column to summarize: ").strip()
            ci = col_index(headers, col_name)
            if ci < 0:
                print("  Column not found.")
                continue
            summarize_col(work_rows, ci, headers[ci])

        elif choice == "5":
            if not headers:
                print("  No file loaded.")
                continue
            out_str = input("  Output CSV path: ").strip().strip('"')
            if not out_str:
                continue
            save_csv(Path(out_str), headers, work_rows)
            print(f"  Saved {len(work_rows)} rows to {out_str}")

        elif choice == "6":
            path_str = input("  CSV file path: ").strip().strip('"')
            if path_str and load(Path(path_str)):
                work_rows = list(rows)

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
