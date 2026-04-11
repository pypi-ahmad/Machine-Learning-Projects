#!/usr/bin/env python3
"""
run_standalone.py — Run all non-pipeline (standalone) notebook families.

Handles GenAI, RAG, Advance RAG, AgenticAI, Advanced agentic,
Agents with tools and MCP, CrewAI, LangGraph, LlamaIndex, Fine tuning,
100_Local_AI_Projects, Conceptual — all using nbconvert.

Usage:
    python run_standalone.py                   # run all
    python run_standalone.py --family GenAI    # specific category
    python run_standalone.py --resume          # skip already-ok
    python run_standalone.py --timeout 600     # per-notebook timeout
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS_FILE = ROOT / "_standalone_results.json"
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = Path(sys.executable)

EXCLUDE_PATTERNS = {
    "venv", ".venv", "Source Code", "__pycache__", ".git", ".github",
    ".ipynb_checkpoints", "_checkpoints", "catboost_info", "artifacts",
    "outputs", "logs", "reports", "tests", "scripts", "shared", 
    "tools", "utils", "configs", "config", "benchmarks",
}

# Standalone families (no pipeline.py)
STANDALONE_FAMILIES = [
    "GenAI",
    "RAG",
    "Advance RAG",
    "AgenticAI",
    "Advanced agentic",
    "Agents with tools and MCP",
    "CrewAI",
    "LangGraph",
    "LlamaIndex",
    "Fine tuning",
    "100_Local_AI_Projects",
    "Conceptual",
]


def discover_standalone_notebooks(family_filter=""):
    """Find standalone notebooks not adjacent to pipeline.py."""
    nbs = []
    for nb_path in sorted(ROOT.rglob("*.ipynb")):
        # Skip excluded dirs
        parts_set = set(nb_path.parts)
        skip = False
        for part in nb_path.parts:
            if any(exc.lower() == part.lower() for exc in EXCLUDE_PATTERNS):
                skip = True
                break
        if skip:
            continue
        # Skip if adjacent to pipeline.py (handled by run_notebooks.py)
        if (nb_path.parent / "pipeline.py").exists():
            continue
        # Skip checkpoint and output files
        if any(x in nb_path.name for x in ["_out_test", "executed_check", "checkpoint"]):
            continue
        # Check family membership
        path_str = str(nb_path)
        is_standalone_family = any(
            fam.lower() in path_str.lower() for fam in STANDALONE_FAMILIES
        )
        if not is_standalone_family:
            continue
        # Apply filter
        if family_filter and family_filter.lower() not in path_str.lower():
            continue
        nbs.append(nb_path)
    return nbs


def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text("utf-8"))
    return {}


def save_results(results):
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), "utf-8")


def get_cell_errors(nb_path: Path) -> list:
    """Return list of (cell_idx, ename, evalue) for error outputs."""
    try:
        nb = json.loads(nb_path.read_text("utf-8", errors="replace"))
        return [
            (i, o.get("ename", "?"), o.get("evalue", "")[:150])
            for i, c in enumerate(nb.get("cells", []))
            for o in c.get("outputs", [])
            if o.get("output_type") == "error"
        ]
    except Exception:
        return []


def run_notebook(nb_path: Path, timeout: int) -> dict:
    """Execute a notebook with nbconvert, return result dict."""
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(VENV_PY), "-m", "jupyter", "nbconvert",
             "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.kernel_name=python3",
             f"--ExecutePreprocessor.timeout={timeout}",
             str(nb_path)],
            capture_output=True,
            text=True,
            timeout=timeout + 90,
            cwd=str(nb_path.parent),
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        elapsed = round(time.perf_counter() - t0, 1)
        if proc.returncode == 0:
            errs = get_cell_errors(nb_path)
            if errs:
                return {"status": "cell_error", "time_s": elapsed,
                        "error": f"Cell errors: {errs[:2]}"}
            return {"status": "ok", "time_s": elapsed, "error": None}
        # Extract last meaningful error line
        stderr = proc.stderr or ""
        err_lines = [l.strip() for l in stderr.splitlines()
                     if l.strip() and not l.startswith("E ") == False
                     and "MissingIDFieldWarning" not in l
                     and "warnings.warn" not in l]
        msg = err_lines[-1][:300] if err_lines else f"exit code {proc.returncode}"
        return {"status": "error", "time_s": elapsed, "error": msg}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "time_s": timeout, "error": f"Timeout after {timeout}s"}
    except Exception as exc:
        return {"status": "error", "time_s": 0, "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    nbs = discover_standalone_notebooks(args.family)
    print(f"Discovered {len(nbs)} standalone notebooks")
    if args.family:
        print(f"Filtered to: {args.family}")

    results = load_results() if args.resume else {}
    ok = fail = skip = 0

    for i, nb_path in enumerate(nbs, 1):
        key = nb_path.relative_to(ROOT).as_posix()
        if args.resume and results.get(key, {}).get("status") == "ok":
            skip += 1
            continue

        family_name = nb_path.parent.parent.name
        print(f"\n[{i}/{len(nbs)}] [{family_name}] {nb_path.name}")

        result = run_notebook(nb_path, args.timeout)
        results[key] = result

        if result["status"] == "ok":
            ok += 1
            print(f"  ✓ OK ({result['time_s']}s)")
        else:
            fail += 1
            err = (result.get("error") or "")[:120]
            print(f"  ✗ {result['status'].upper()} ({result['time_s']}s): {err}")

        save_results(results)

    print(f"\n{'='*65}")
    print(f"STANDALONE RESULTS: {ok} OK  |  {fail} FAIL  |  {skip} SKIPPED")
    print(f"{'='*65}")

    if fail:
        print("\nFailed notebooks:")
        for key, res in sorted(results.items()):
            if res.get("status") != "ok":
                print(f"  [{res['status']}] {key}: {(res.get('error') or '')[:80]}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
