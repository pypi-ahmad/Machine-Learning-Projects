"""Environment Variable Manager — CLI tool.

View, search, add (session-only), and export environment variables.
Groups variables by common prefixes (PATH, PYTHON, etc.) and
shows path entries split line-by-line.

Usage:
    python main.py
"""

import os
import platform
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def all_vars() -> dict[str, str]:
    return dict(os.environ)


def search_vars(keyword: str, search_values: bool = False) -> list[tuple[str, str]]:
    kw = keyword.lower()
    result = []
    for k, v in os.environ.items():
        if kw in k.lower() or (search_values and kw in v.lower()):
            result.append((k, v))
    return sorted(result)


def group_by_prefix() -> dict[str, list[tuple[str, str]]]:
    """Group env vars by common leading prefix."""
    from collections import defaultdict
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for k, v in sorted(os.environ.items()):
        if "_" in k:
            prefix = k.split("_")[0]
        elif k[:4].isupper() and len(k) > 4:
            prefix = k[:4]
        else:
            prefix = k
        groups[prefix].append((k, v))
    return dict(groups)


def set_var(key: str, value: str) -> None:
    """Set env variable for the current process (session only)."""
    os.environ[key] = value


def unset_var(key: str) -> bool:
    if key in os.environ:
        del os.environ[key]
        return True
    return False


def export_shell(out_path: Path) -> None:
    """Export as shell script."""
    lines = ["#!/bin/sh\n"]
    for k, v in sorted(os.environ.items()):
        safe_v = v.replace("'", "'\\''")
        lines.append(f"export {k}='{safe_v}'\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def export_dotenv(out_path: Path) -> None:
    """Export as .env file."""
    lines = []
    for k, v in sorted(os.environ.items()):
        # Skip values with newlines for .env format
        if "\n" not in v:
            safe_v = v.replace('"', '\\"')
            lines.append(f'{k}="{safe_v}"\n')
    out_path.write_text("".join(lines), encoding="utf-8")


def path_entries() -> list[str]:
    """Return PATH entries as a list."""
    sep = ";" if platform.system() == "Windows" else ":"
    return os.environ.get("PATH", "").split(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Environment Variable Manager
-----------------------------
1. List all variables
2. Search by key or value
3. Show by group/prefix
4. Show PATH entries
5. Set variable (this session)
6. Unset variable (this session)
7. Export to file (.env / shell)
0. Quit
"""


def print_var(k: str, v: str, max_val: int = 80) -> None:
    display = v.replace("\n", "\\n").replace("\r", "\\r")
    print(f"  {k:<35} = {display[:max_val]}")


def main() -> None:
    print("Environment Variable Manager")
    print(f"  Platform: {platform.system()}  |  {len(os.environ)} variables loaded")

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            vars_all = all_vars()
            n_s = input("  Max to show (default 50, 0=all): ").strip()
            n = int(n_s) if n_s.isdigit() else 50
            n = len(vars_all) if n == 0 else n
            print(f"\n  {len(vars_all)} environment variable(s):")
            for i, (k, v) in enumerate(sorted(vars_all.items())):
                if i >= n:
                    print(f"  ... {len(vars_all) - n} more")
                    break
                print_var(k, v)

        elif choice == "2":
            kw = input("  Keyword: ").strip()
            if not kw:
                continue
            sv = input("  Search values too? (y/n, default n): ").strip().lower() == "y"
            results = search_vars(kw, sv)
            if results:
                print(f"\n  {len(results)} match(es):")
                for k, v in results:
                    print_var(k, v)
            else:
                print(f"  No matches for '{kw}'.")

        elif choice == "3":
            groups = group_by_prefix()
            print(f"\n  {len(groups)} groups:")
            for prefix, items in sorted(groups.items()):
                print(f"\n  [{prefix}]  ({len(items)} var(s))")
                for k, v in items:
                    print_var(k, v)

        elif choice == "4":
            entries = path_entries()
            print(f"\n  PATH  ({len(entries)} entries):")
            for i, entry in enumerate(entries, 1):
                exists = Path(entry).exists() if entry else False
                status = "\033[32m✓\033[0m" if exists else "\033[31m✗\033[0m"
                print(f"  {i:>3}. {status} {entry}")

        elif choice == "5":
            key = input("  Variable name: ").strip().upper()
            if not key:
                continue
            current = os.environ.get(key)
            if current:
                print(f"  Current: {current[:80]}")
            value = input("  New value: ")
            set_var(key, value)
            print(f"  Set {key} = {value[:80]} (this session only)")

        elif choice == "6":
            key = input("  Variable name: ").strip().upper()
            if not key:
                continue
            ok = unset_var(key)
            if ok:
                print(f"  Unset {key} (this session only)")
            else:
                print(f"  Variable '{key}' not found.")

        elif choice == "7":
            fmt = input("  Format: (d)otenv or (s)hell script? (default d): ").strip().lower()
            ext = ".sh" if fmt.startswith("s") else ".env"
            out_s = input(f"  Output file (default vars{ext}): ").strip() or f"vars{ext}"
            out_path = Path(out_s)
            if fmt.startswith("s"):
                export_shell(out_path)
            else:
                export_dotenv(out_path)
            print(f"  Exported {len(os.environ)} variables to {out_path}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
