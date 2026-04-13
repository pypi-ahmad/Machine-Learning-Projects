"""JSON Formatter — CLI tool.

Pretty-print, minify, validate, sort keys, diff two JSON objects,
flatten/unflatten nested JSON, and query values with dot-notation paths.

Usage:
    python main.py
    python main.py data.json
"""

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def pretty_print(obj, indent: int = 2, sort_keys: bool = False) -> str:
    return json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=False)


def minify(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def validate(text: str) -> tuple[bool, str]:
    try:
        json.loads(text)
        return True, "Valid JSON."
    except json.JSONDecodeError as e:
        return False, str(e)


def flatten(obj, prefix: str = "", sep: str = ".") -> dict:
    """Flatten nested JSON to dot-notation keys."""
    items: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}{sep}{k}" if prefix else k
            items.update(flatten(v, new_key, sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{prefix}{sep}{i}" if prefix else str(i)
            items.update(flatten(v, new_key, sep))
    else:
        items[prefix] = obj
    return items


def unflatten(flat: dict, sep: str = ".") -> dict:
    """Reconstruct nested dict from flat dot-notation dict."""
    result: dict = {}
    for key, value in flat.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result


def query(obj, path: str):
    """Get value at dot-notation path; returns None if missing."""
    parts = path.split(".")
    cur = obj
    for part in parts:
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def diff_json(a, b, path: str = "") -> list[str]:
    """Simple structural diff between two JSON objects."""
    diffs = []
    if type(a) != type(b):
        diffs.append(f"  {path or '(root)'}: type {type(a).__name__} → {type(b).__name__}")
        return diffs
    if isinstance(a, dict):
        all_keys = set(a) | set(b)
        for k in sorted(all_keys):
            p = f"{path}.{k}" if path else k
            if k not in a:
                diffs.append(f"  + {p}: {json.dumps(b[k])}")
            elif k not in b:
                diffs.append(f"  - {p}: {json.dumps(a[k])}")
            else:
                diffs.extend(diff_json(a[k], b[k], p))
    elif isinstance(a, list):
        for i in range(max(len(a), len(b))):
            p = f"{path}[{i}]"
            if i >= len(a):
                diffs.append(f"  + {p}: {json.dumps(b[i])}")
            elif i >= len(b):
                diffs.append(f"  - {p}: {json.dumps(a[i])}")
            else:
                diffs.extend(diff_json(a[i], b[i], p))
    else:
        if a != b:
            diffs.append(f"  ~ {path or '(root)'}: {json.dumps(a)} → {json.dumps(b)}")
    return diffs


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def read_json_input(label: str = "JSON"):
    print(f"  Enter {label} (type '###' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line == "###":
            break
        lines.append(line)
    text = "\n".join(lines)
    try:
        return json.loads(text), text
    except json.JSONDecodeError as e:
        print(f"  Parse error: {e}")
        return None, text


def load_file(prompt: str):
    path_str = input(prompt).strip().strip('"')
    p = Path(path_str)
    if not p.exists():
        print(f"  File not found: {path_str}")
        return None, None
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text), text
    except json.JSONDecodeError as e:
        print(f"  Parse error: {e}")
        return None, text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
JSON Formatter
--------------
1. Validate JSON
2. Pretty-print JSON
3. Minify JSON
4. Sort keys
5. Flatten (nested → dot-keys)
6. Query by path
7. Diff two JSON objects
8. Format a JSON file
0. Quit
"""


def main() -> None:
    # Direct file argument
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            text = p.read_text(encoding="utf-8")
            try:
                obj = json.loads(text)
                print(pretty_print(obj))
            except json.JSONDecodeError as e:
                print(f"JSON error: {e}", file=sys.stderr)
                sys.exit(1)
            return

    print("JSON Formatter")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            _, text = read_json_input()
            if text:
                ok, msg = validate(text)
                icon = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
                print(f"\n  {icon} {msg}")

        elif choice == "2":
            obj, _ = read_json_input()
            if obj is not None:
                indent_s = input("  Indent (default 2): ").strip()
                indent = int(indent_s) if indent_s.isdigit() else 2
                print(f"\n{pretty_print(obj, indent)}")

        elif choice == "3":
            obj, _ = read_json_input()
            if obj is not None:
                print(f"\n  {minify(obj)}")

        elif choice == "4":
            obj, _ = read_json_input()
            if obj is not None:
                print(f"\n{pretty_print(obj, sort_keys=True)}")

        elif choice == "5":
            obj, _ = read_json_input()
            if obj is not None:
                flat = flatten(obj)
                print(f"\n  {len(flat)} flattened key(s):")
                for k, v in list(flat.items())[:50]:
                    print(f"    {k}: {json.dumps(v)}")
                if len(flat) > 50:
                    print(f"  ... and {len(flat) - 50} more")

        elif choice == "6":
            obj, _ = read_json_input()
            if obj is not None:
                path = input("  Path (e.g. user.address.city): ").strip()
                val  = query(obj, path)
                print(f"\n  Result: {json.dumps(val, indent=2)}")

        elif choice == "7":
            print("  First JSON:")
            obj_a, _ = read_json_input("JSON A")
            print("  Second JSON:")
            obj_b, _ = read_json_input("JSON B")
            if obj_a is not None and obj_b is not None:
                diffs = diff_json(obj_a, obj_b)
                if diffs:
                    print(f"\n  {len(diffs)} difference(s):")
                    for d in diffs:
                        print(d)
                else:
                    print("\n  Objects are identical.")

        elif choice == "8":
            obj, _ = load_file("  JSON file path: ")
            if obj is not None:
                out = pretty_print(obj)
                save = input("  Save formatted output? (y/n, default n): ").strip().lower()
                print(f"\n{out}")
                if save == "y":
                    path_str = input("  Output file (blank = overwrite): ").strip().strip('"')
                    out_p = Path(path_str) if path_str else Path(
                        input("  Original file: ").strip().strip('"'))
                    out_p.write_text(out, encoding="utf-8")
                    print(f"  Saved to: {out_p}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
