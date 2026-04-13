"""JSON to Dataclass — CLI developer tool.

Convert JSON objects to Python dataclass, Pydantic model,
TypedDict, or attrs class definitions.

Usage:
    python main.py
    python main.py --file data.json --output models.py
    python main.py --json '{"name":"Alice","age":30}'
    python main.py --url https://api.example.com/data
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from typing import Any


ANSI = {"bold": "\033[1m", "cyan": "\033[96m", "green": "\033[92m",
        "yellow": "\033[93m", "red": "\033[91m", "reset": "\033[0m"}


def c(text, color):
    return f"{ANSI.get(color,'')}{text}{ANSI['reset']}"


# ── Type inference ─────────────────────────────────────────────────────────────

def infer_type(value: Any, key: str = "value") -> str:
    if value is None:
        return "Optional[Any]"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        # Heuristic sub-types
        if re.match(r"\d{4}-\d{2}-\d{2}T", value):
            return "datetime"
        if re.match(r"\d{4}-\d{2}-\d{2}$", value):
            return "date"
        if re.match(r"https?://", value):
            return "str  # URL"
        if "@" in value and "." in value:
            return "str  # email"
        return "str"
    if isinstance(value, list):
        if not value:
            return "list[Any]"
        elem_type = infer_type(value[0])
        return f"list[{elem_type}]"
    if isinstance(value, dict):
        return to_pascal(key)
    return "Any"


def to_pascal(name: str) -> str:
    """Convert snake_case or camelCase to PascalCase."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    parts = re.split(r"[_\s]+", re.sub(r"([A-Z])", r"_\1", name).lstrip("_"))
    return "".join(p.capitalize() for p in parts if p) or "Model"


def to_snake(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


# ── Code generators ───────────────────────────────────────────────────────────

def collect_classes(obj: dict, class_name: str, classes: dict):
    """Recursively collect nested classes."""
    fields = {}
    for k, v in obj.items():
        snake_k    = to_snake(k)
        field_type = infer_type(v, k)
        fields[snake_k] = (field_type, v)
        if isinstance(v, dict):
            nested = to_pascal(k)
            collect_classes(v, nested, classes)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            nested = to_pascal(k)
            collect_classes(v[0], nested, classes)
    classes[class_name] = fields


def gen_dataclass(obj: dict, class_name: str = "Model") -> str:
    classes: dict = {}
    collect_classes(obj, class_name, classes)

    imports = [
        "from __future__ import annotations",
        "from dataclasses import dataclass, field",
        "from datetime import date, datetime",
        "from typing import Any, Optional",
        "",
    ]
    lines = list(imports)

    # Emit nested classes first (leaf classes first)
    ordered = list(reversed(list(classes.keys())))
    for cname in ordered:
        fields = classes[cname]
        lines.append("@dataclass")
        lines.append(f"class {cname}:")
        if not fields:
            lines.append("    pass")
        for fname, (ftype, fval) in fields.items():
            default = ""
            if fval is None:
                default = " = None"
            elif isinstance(fval, bool):
                default = f" = {fval}"
            elif isinstance(fval, (int, float)):
                default = f" = {fval}"
            elif isinstance(fval, str):
                esc = fval.replace('"', '\\"')[:40]
                default = f' = "{esc}"'
            elif isinstance(fval, list):
                default = " = field(default_factory=list)"
            lines.append(f"    {fname}: {ftype}{default}")
        lines.append("")

    return "\n".join(lines)


def gen_pydantic(obj: dict, class_name: str = "Model") -> str:
    classes: dict = {}
    collect_classes(obj, class_name, classes)

    imports = [
        "from __future__ import annotations",
        "from datetime import date, datetime",
        "from typing import Any, List, Optional",
        "from pydantic import BaseModel",
        "",
    ]
    lines = list(imports)
    ordered = list(reversed(list(classes.keys())))
    for cname in ordered:
        fields = classes[cname]
        lines.append(f"class {cname}(BaseModel):")
        if not fields:
            lines.append("    pass")
        for fname, (ftype, fval) in fields.items():
            if fval is None:
                lines.append(f"    {fname}: Optional[{ftype}] = None")
            else:
                lines.append(f"    {fname}: {ftype}")
        lines.append("")

    return "\n".join(lines)


def gen_typeddict(obj: dict, class_name: str = "Model") -> str:
    classes: dict = {}
    collect_classes(obj, class_name, classes)

    imports = [
        "from __future__ import annotations",
        "from datetime import date, datetime",
        "from typing import Any, List, Optional, TypedDict",
        "",
    ]
    lines = list(imports)
    ordered = list(reversed(list(classes.keys())))
    for cname in ordered:
        fields = classes[cname]
        lines.append(f"class {cname}(TypedDict):")
        if not fields:
            lines.append("    pass")
        for fname, (ftype, _) in fields.items():
            lines.append(f"    {fname}: {ftype}")
        lines.append("")

    return "\n".join(lines)


def gen_attrs(obj: dict, class_name: str = "Model") -> str:
    classes: dict = {}
    collect_classes(obj, class_name, classes)

    imports = [
        "from __future__ import annotations",
        "from datetime import date, datetime",
        "from typing import Any, List, Optional",
        "import attr",
        "",
    ]
    lines = list(imports)
    ordered = list(reversed(list(classes.keys())))
    for cname in ordered:
        fields = classes[cname]
        lines.append("@attr.s(auto_attribs=True)")
        lines.append(f"class {cname}:")
        if not fields:
            lines.append("    pass")
        for fname, (ftype, fval) in fields.items():
            if fval is None:
                lines.append(f"    {fname}: Optional[{ftype}] = None")
            elif isinstance(fval, list):
                lines.append(f"    {fname}: {ftype} = attr.Factory(list)")
            else:
                lines.append(f"    {fname}: {ftype}")
        lines.append("")

    return "\n".join(lines)


GENERATORS = {
    "dataclass": gen_dataclass,
    "pydantic":  gen_pydantic,
    "typeddict": gen_typeddict,
    "attrs":     gen_attrs,
}


def process(data: Any, class_name: str = "Model", style: str = "dataclass") -> str:
    if isinstance(data, list):
        if not data or not isinstance(data[0], dict):
            return f"# Top-level array of primitives\nfrom typing import list\nItems = list[{infer_type(data[0] if data else None)}]\n"
        data = data[0]   # use first element as schema
    if not isinstance(data, dict):
        return f"# Not a JSON object (type: {type(data).__name__})\n"
    fn = GENERATORS.get(style, gen_dataclass)
    return fn(data, class_name)


def interactive_mode():
    print(c("JSON to Dataclass Converter\n", "bold"))
    print("Commands: convert, file <path>, url <url>, quit")
    print("Styles:   dataclass (default), pydantic, typeddict, attrs\n")

    while True:
        try:
            line = input(c("j2d> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break

        parts = line.split()
        cmd   = parts[0].lower()

        if cmd == "convert":
            print("Paste JSON (type END on a new line to finish):")
            lines = []
            while True:
                try:
                    l = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if l.strip() == "END":
                    break
                lines.append(l)
            try:
                data   = json.loads("\n".join(lines))
                name   = input(c("  Class name [Model]: ", "cyan")).strip() or "Model"
                style  = input(c("  Style (dataclass/pydantic/typeddict/attrs) [dataclass]: ", "cyan")).strip() or "dataclass"
                result = process(data, to_pascal(name), style)
                print(c("\n─── Generated Code ───────────────────────", "dim"))
                print(result)
            except json.JSONDecodeError as e:
                print(c(f"  JSON parse error: {e}", "red"))

        elif cmd == "file" and len(parts) > 1:
            try:
                with open(parts[1]) as f:
                    data = json.load(f)
                name   = to_pascal(os.path.splitext(os.path.basename(parts[1]))[0])
                result = process(data, name)
                print(result)
            except Exception as e:
                print(c(f"  Error: {e}", "red"))

        elif cmd == "url" and len(parts) > 1:
            try:
                with urllib.request.urlopen(parts[1], timeout=8) as resp:
                    data = json.loads(resp.read())
                result = process(data)
                print(result)
            except Exception as e:
                print(c(f"  Error: {e}", "red"))

        else:
            # Try treating as inline JSON
            try:
                data   = json.loads(line)
                result = process(data)
                print(result)
            except json.JSONDecodeError:
                print(c("  Unknown command or invalid JSON.", "yellow"))


def main():
    parser = argparse.ArgumentParser(description="Convert JSON to Python type definitions")
    parser.add_argument("--file",   metavar="FILE",    help="Input JSON file")
    parser.add_argument("--json",   metavar="JSON",    help="Inline JSON string")
    parser.add_argument("--url",    metavar="URL",     help="Fetch JSON from URL")
    parser.add_argument("--name",   metavar="NAME",    default="Model")
    parser.add_argument("--style",  metavar="STYLE",   default="dataclass",
                        choices=list(GENERATORS.keys()))
    parser.add_argument("--output", metavar="FILE",    help="Output Python file")
    args = parser.parse_args()

    data = None
    if args.file:
        with open(args.file) as f:
            data = json.load(f)
    elif args.json:
        data = json.loads(args.json)
    elif args.url:
        with urllib.request.urlopen(args.url, timeout=8) as resp:
            data = json.loads(resp.read())

    if data is not None:
        result = process(data, to_pascal(args.name), args.style)
        if args.output:
            with open(args.output, "w") as f:
                f.write(result)
            print(c(f"✓ Written to {args.output}", "green"))
        else:
            print(result)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
