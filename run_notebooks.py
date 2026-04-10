#!/usr/bin/env python3
"""
run_notebooks.py — Execute all generated .ipynb notebooks programmatically.

Uses papermill for execution with timeout handling. Each notebook is run
in its own working directory so that __file__ and save paths resolve correctly.

Usage:
    python run_notebooks.py                      # run all
    python run_notebooks.py --pilot 5            # run first 5
    python run_notebooks.py --family clustering  # run matching family
    python run_notebooks.py --resume             # skip already-done
    python run_notebooks.py --timeout 600        # 10-min per notebook
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import nbformat
import papermill as pm

EXCLUDE_DIRS = {"venv", ".venv", "core", "data", "__pycache__", ".git", ".github"}
ROOT = Path(__file__).parent
RESULTS_FILE = ROOT / "_notebook_results.json"


def discover_notebooks():
    """Find all generated notebooks sitting beside a pipeline.py."""
    nbs = []
    for p in sorted(ROOT.rglob("pipeline.py")):
        parts_lower = {x.lower() for x in p.parts}
        if parts_lower & {d.lower() for d in EXCLUDE_DIRS}:
            continue
        if any("data analysis" in x.lower() for x in p.parts):
            continue
        safe_name = re.sub(r'[<>:"|?*]', "_", p.parent.name)
        nb_path = p.parent / f"{safe_name}.ipynb"
        if nb_path.exists():
            nbs.append(nb_path)
    return nbs


def load_results():
    """Load previous results for --resume."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text("utf-8"))
    return {}


def save_results(results):
    """Persist execution results."""
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), "utf-8")


def run_one(nb_path: Path, timeout: int = 900) -> dict:
    """Execute a single notebook. Returns result dict."""
    project_dir = nb_path.parent
    out_path = nb_path  # overwrite in-place to preserve outputs

    t0 = time.perf_counter()
    try:
        pm.execute_notebook(
            str(nb_path),
            str(out_path),
            cwd=str(project_dir),
            kernel_name="python3",
            timeout=timeout,
            progress_bar=False,
        )
        elapsed = time.perf_counter() - t0
        return {
            "status": "ok",
            "time_s": round(elapsed, 1),
            "error": None,
        }
    except pm.PapermillExecutionError as exc:
        elapsed = time.perf_counter() - t0
        return {
            "status": "error",
            "time_s": round(elapsed, 1),
            "error": f"Cell {exc.cell_index}: {exc.ename}: {exc.evalue}",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "status": "error",
            "time_s": round(elapsed, 1),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser(description="Execute all generated notebooks")
    parser.add_argument("--pilot", type=int, default=0, help="Run only first N notebooks")
    parser.add_argument("--family", type=str, default="", help="Filter by family name (substring match)")
    parser.add_argument("--resume", action="store_true", help="Skip already-successful notebooks")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per notebook (seconds)")
    parser.add_argument("--start", type=int, default=0, help="Start from Nth notebook (0-based)")
    args = parser.parse_args()

    nbs = discover_notebooks()
    print(f"Discovered {len(nbs)} notebooks\n")

    if args.family:
        nbs = [n for n in nbs if args.family.lower() in str(n).lower()]
        print(f"Filtered to {len(nbs)} notebooks matching '{args.family}'")

    if args.start > 0:
        nbs = nbs[args.start:]
        print(f"Starting from index {args.start}, {len(nbs)} remaining")

    if args.pilot > 0:
        nbs = nbs[:args.pilot]
        print(f"Pilot mode: running {len(nbs)} notebooks")

    results = load_results() if args.resume else {}
    ok = fail = skip = 0

    for i, nb_path in enumerate(nbs):
        key = nb_path.relative_to(ROOT).as_posix()
        if args.resume and key in results and results[key].get("status") == "ok":
            skip += 1
            continue

        print(f"\n[{i+1}/{len(nbs)}] {key}")
        result = run_one(nb_path, args.timeout)
        results[key] = result

        if result["status"] == "ok":
            ok += 1
            print(f"  OK ({result['time_s']}s)")
        else:
            fail += 1
            print(f"  FAIL ({result['time_s']}s): {result['error']}")

        # Save after each notebook (crash-resilient)
        save_results(results)

    print(f"\n{'='*60}")
    print(f"RESULTS: {ok} OK, {fail} FAIL, {skip} SKIPPED")
    print(f"{'='*60}")

    if fail:
        print("\nFailed notebooks:")
        for key, res in results.items():
            if res.get("status") == "error":
                print(f"  {key}: {res['error']}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
