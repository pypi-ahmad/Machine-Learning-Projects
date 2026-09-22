"""Print lines shared by two text files, preserving the first file's order."""

import argparse
from pathlib import Path


DEFAULT_FIRST = Path(__file__).with_name("1.txt")
DEFAULT_SECOND = Path(__file__).with_name("2.txt")


def common_lines(first: Path, second: Path) -> list[str]:
    """Return unique non-blank lines found in both files, in first-file order."""
    second_lines = set(second.read_text(encoding="utf-8").splitlines())
    shared = []
    for line in first.read_text(encoding="utf-8").splitlines():
        if line and line in second_lines and line not in shared:
            shared.append(line)
    return shared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", nargs="?", type=Path, default=DEFAULT_FIRST)
    parser.add_argument("second", nargs="?", type=Path, default=DEFAULT_SECOND)
    parser.add_argument("--output", "-o", type=Path, help="Write shared lines to this new file.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.first.is_file() or not args.second.is_file():
        raise SystemExit("Both inputs must be existing regular files.")
    if args.output and args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {args.output}. Re-run with --overwrite.")

    shared = common_lines(args.first, args.second)
    if args.output:
        args.output.write_text("\n".join(shared) + ("\n" if shared else ""), encoding="utf-8")
        print(f"Wrote {len(shared)} shared line(s) to {args.output}.")
        return

    print(f"{len(shared)} shared line(s):")
    for line in shared:
        print(line)


if __name__ == "__main__":
    main()
