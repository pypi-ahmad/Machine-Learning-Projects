#!/usr/bin/env python3
"""
generate_notebooks.py
=====================
Convert every pipeline.py into a structured Jupyter Notebook (.ipynb).

Strategy
--------
1. Use Python's ``ast`` module to parse each pipeline.py and locate the
   module docstring, import zone, constant zone, function/class definitions,
   and the ``if __name__ == "__main__"`` guard.
2. Build notebook cells:
   - Title & overview (markdown)
   - Environment setup (__file__, %matplotlib inline)
   - Imports & configuration
   - One cell per function/class definition
   - Execute cell (``main()``)
3. Write the notebook JSON beside the original pipeline.py.
4. Validate every code cell compiles without syntax errors.
"""
import ast
import json
import re
import sys
import textwrap
from pathlib import Path

EXCLUDE_DIRS = {
    "venv", ".venv", "core", "data", "__pycache__", ".git", ".github",
}


# ── Notebook helpers ──────────────────────────────────────────────────────
def _cell(cell_type: str, source: str):
    """Return a single notebook cell dict."""
    if not source.endswith("\n"):
        source += "\n"
    src_lines = source.splitlines(True)
    cell = {"cell_type": cell_type, "metadata": {}, "source": src_lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _notebook(cells: list) -> dict:
    """Wrap cells in a complete nbformat-4 document."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.0",
            },
        },
        "cells": cells,
    }


# ── Conversion logic ─────────────────────────────────────────────────────
def _convert(pipeline_path: Path):
    """Parse *pipeline_path* and return a notebook dict (or None on error)."""
    src = pipeline_path.read_text("utf-8", errors="ignore")
    lines = src.splitlines(True)
    n = len(lines)

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    top = list(ast.iter_child_nodes(tree))
    project_name = pipeline_path.parent.name

    # ── 1. Module docstring ──────────────────────────────────────────
    mod_doc = ast.get_docstring(tree) or ""
    doc_end = 0                           # 0-based exclusive line
    if (
        top
        and isinstance(top[0], ast.Expr)
        and isinstance(getattr(top[0], "value", None), ast.Constant)
        and isinstance(top[0].value.value, str)
    ):
        doc_end = top[0].end_lineno       # 1-indexed → 0-based exclusive

    # ── 2. Locate function / class definitions ───────────────────────
    defs = []          # (start_0, end_0, node)
    first_def = n      # line index of the earliest def/class
    has_main = False

    for node in top:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s = node.lineno - 1
            e = node.end_lineno
            # Include decorators
            if getattr(node, "decorator_list", []):
                s = min(s, node.decorator_list[0].lineno - 1)
            defs.append((s, e, node))
            first_def = min(first_def, s)
            if getattr(node, "name", "") == "main":
                has_main = True

    # ── 3. Preamble (imports + constants) ────────────────────────────
    preamble = "".join(lines[doc_end:first_def]).strip()

    # Replace matplotlib Agg backend – notebooks use %matplotlib inline
    preamble = re.sub(
        r"import\s+matplotlib\s*;\s*matplotlib\.use\(\s*[\"']Agg[\"']\s*\)",
        "import matplotlib",
        preamble,
    )
    preamble = re.sub(
        r"matplotlib\.use\(\s*[\"']Agg[\"']\s*\)",
        "# (backend set by %matplotlib inline)",
        preamble,
    )

    # ── Build cells ──────────────────────────────────────────────────
    cells = []

    # A — Title
    title = f"# {project_name}\n"
    if mod_doc:
        title += f"\n{mod_doc}\n"
    cells.append(_cell("markdown", title))

    # B — Notebook setup
    cells.append(_cell("markdown", "## Environment Setup\n"))
    cells.append(_cell("code", textwrap.dedent("""\
        import os
        # Define __file__ so save-path logic in the pipeline works correctly
        __file__ = os.path.abspath('pipeline.py')
        %matplotlib inline
    """)))

    # C — Imports & configuration
    if preamble:
        cells.append(_cell("markdown", "## Imports & Configuration\n"))
        cells.append(_cell("code", preamble + "\n"))

    # D — Function / class cells
    last_end = first_def
    for s, e, node in defs:
        # Extend start backwards to capture preceding comments
        cs = s
        while cs > last_end and lines[cs - 1].strip().startswith("#"):
            cs -= 1

        # Module-level gap (rare: code between two defs that isn't a def)
        gap = "".join(lines[last_end:cs]).strip()
        if gap:
            cells.append(_cell("code", gap + "\n"))

        func_src = "".join(lines[cs:e]).rstrip() + "\n"

        name = getattr(node, "name", "")
        if name == "main":
            cells.append(_cell("markdown", "## Main Pipeline\n"))
        else:
            nice = name.replace("_", " ").title()
            fd = ast.get_docstring(node) or ""
            hdr = f"## {nice}\n"
            if fd:
                hdr += f"\n{fd}\n"
            cells.append(_cell("markdown", hdr))

        cells.append(_cell("code", func_src))
        last_end = e

    # E — Execute cell
    cells.append(_cell("markdown",
        "## Execute Pipeline\n\nRun the complete ML pipeline end-to-end:\n"))
    if has_main:
        cells.append(_cell("code", "main()\n"))
    else:
        cells.append(_cell("code",
            "# No main() found — call the appropriate entry-point manually\n"
            "print('Pipeline loaded. Call functions as needed.')\n"))

    return _notebook(cells)


def _validate_nb(nb: dict, label: str) -> list[str]:
    """Compile every code cell; return a list of error messages."""
    errors = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        # Strip IPython magics before compiling
        clean = "\n".join(
            l for l in code.splitlines()
            if not l.strip().startswith("%") and not l.strip().startswith("!")
        )
        if not clean.strip():
            continue
        try:
            compile(clean, f"<{label}:cell{i}>", "exec")
        except SyntaxError as exc:
            errors.append(f"  cell {i}: {exc}")
    return errors


# ── Main driver ──────────────────────────────────────────────────────────
def main():
    root = Path(".")
    exclude = {x.lower() for x in EXCLUDE_DIRS}

    pipelines = []
    for p in sorted(root.rglob("pipeline.py")):
        parts_lower = {x.lower() for x in p.parts}
        if parts_lower & exclude:
            continue
        if any("data analysis" in x.lower() for x in p.parts):
            continue
        pipelines.append(p)

    print(f"Found {len(pipelines)} pipeline.py files\n")

    ok = skip = errs = 0
    all_errors = []

    for p in pipelines:
        nb = _convert(p)
        if nb is None:
            print(f"  SKIP (syntax error): {p}")
            skip += 1
            continue

        # Validate
        ve = _validate_nb(nb, p.as_posix())
        if ve:
            all_errors.append((p.as_posix(), ve))
            errs += 1

        # Write notebook beside pipeline.py
        safe_name = re.sub(r'[<>:"|?*]', "_", p.parent.name)
        out_path = p.parent / f"{safe_name}.ipynb"
        out_path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        ok += 1

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\nGenerated : {ok}")
    print(f"Skipped   : {skip}")
    print(f"Validation errors: {errs}")
    if all_errors:
        print("\n── Validation issues ──")
        for path, msgs in all_errors:
            print(f"\n{path}:")
            for m in msgs:
                print(m)

    return 0 if (errs == 0 and skip == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
