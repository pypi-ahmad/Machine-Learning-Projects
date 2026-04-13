"""CSV Cleaner — CLI tool.

Clean a CSV file by:
  • Removing duplicate rows
  • Dropping columns with too many missing values
  • Filling or dropping rows with missing values
  • Stripping whitespace from values
  • Standardising column name casing
  • Removing blank rows

Usage:
    python main.py
    python main.py input.csv
    python main.py input.csv --output clean.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows   = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def save_csv(path: Path, headers: list[str], rows: list[list[str]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def clean(headers: list[str], rows: list[list[str]],
          strip_ws: bool = True,
          dedupe: bool = True,
          drop_blank_rows: bool = True,
          fill_missing: str = "",
          col_missing_threshold: float = 0.5,
          normalize_headers: bool = True) -> tuple[list[str], list[list[str]], dict]:

    stats: dict[str, int] = {}
    original_rows = len(rows)

    # Normalize header names
    if normalize_headers:
        headers = [h.strip().lower().replace(" ", "_") for h in headers]

    # Ensure all rows have same number of columns
    ncols = len(headers)
    rows  = [r + [""] * (ncols - len(r)) if len(r) < ncols else r[:ncols] for r in rows]

    # Strip whitespace
    if strip_ws:
        rows = [[v.strip() for v in r] for r in rows]
        stats["whitespace_stripped"] = 1

    # Drop blank rows (all empty)
    if drop_blank_rows:
        before = len(rows)
        rows   = [r for r in rows if any(v.strip() for v in r)]
        stats["blank_rows_removed"] = before - len(rows)

    # Drop columns with too many missing values
    dropped_cols = []
    if col_missing_threshold < 1.0:
        to_keep = []
        for i, h in enumerate(headers):
            missing = sum(1 for r in rows if not r[i].strip()) / max(len(rows), 1)
            if missing <= col_missing_threshold:
                to_keep.append(i)
            else:
                dropped_cols.append(h)
        if dropped_cols:
            headers = [headers[i] for i in to_keep]
            rows    = [[r[i] for i in to_keep] for r in rows]
    stats["columns_dropped"] = len(dropped_cols)

    # Fill missing values
    if fill_missing != "":
        filled = 0
        for r in rows:
            for i, v in enumerate(r):
                if not v.strip():
                    r[i] = fill_missing
                    filled += 1
        stats["values_filled"] = filled

    # Remove duplicates
    if dedupe:
        seen  = set()
        deduped = []
        for r in rows:
            key = tuple(r)
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        stats["duplicates_removed"] = len(rows) - len(deduped)
        rows = deduped

    stats["original_rows"] = original_rows
    stats["final_rows"]    = len(rows)
    return headers, rows, stats


def main():
    parser = argparse.ArgumentParser(description="CSV Cleaner")
    parser.add_argument("input",  nargs="?")
    parser.add_argument("--output", "-o")
    parser.add_argument("--fill",   default="", help="Fill missing values with this string")
    parser.add_argument("--no-dedupe",       action="store_true")
    parser.add_argument("--no-strip",        action="store_true")
    parser.add_argument("--col-threshold",   type=float, default=0.5,
                        help="Drop columns with more than this fraction missing (default 0.5)")
    args = parser.parse_args()

    if args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        headers, rows = load_csv(path)
        headers, rows, stats = clean(headers, rows,
            strip_ws=not args.no_strip,
            dedupe=not args.no_dedupe,
            fill_missing=args.fill,
            col_missing_threshold=args.col_threshold)
        out = Path(args.output) if args.output else path.with_stem(path.stem + "_clean")
        save_csv(out, headers, rows)
        print(f"  Saved: {out}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    # Interactive
    print("CSV Cleaner")
    print("────────────────────────────")

    while True:
        raw = input("\nCSV file path (or 'q'): ").strip()
        if raw.lower() == "q":
            break
        path = Path(raw)
        if not path.exists():
            print(f"  Not found: {path}")
            continue

        headers, rows = load_csv(path)
        print(f"\n  Loaded: {len(rows)} rows, {len(headers)} columns")
        print(f"  Columns: {', '.join(headers[:8])}{'...' if len(headers)>8 else ''}")

        strip    = input("  Strip whitespace? [Y/n]: ").strip().lower() != "n"
        dedupe   = input("  Remove duplicates? [Y/n]: ").strip().lower() != "n"
        fill     = input("  Fill missing with (blank = leave empty): ").strip()
        thr      = input("  Drop cols with >X% missing [50]: ").strip()
        thr      = float(thr) / 100 if thr else 0.5

        headers, rows, stats = clean(headers, rows,
            strip_ws=strip, dedupe=dedupe,
            fill_missing=fill, col_missing_threshold=thr)

        out = path.with_stem(path.stem + "_clean")
        save_csv(out, headers, rows)
        print(f"\n  Saved: {out}")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
