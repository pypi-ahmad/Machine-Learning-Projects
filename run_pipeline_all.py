#!/usr/bin/env python3
"""
run_pipeline_all.py — Run all pipeline notebook families in sequence.

Runs ALL 318 pipeline notebooks (those adjacent to pipeline.py) across all categories.
Saves results to _notebook_results.json (crash-resilient, resumes from last state).

Usage:
    python run_pipeline_all.py                   # run all
    python run_pipeline_all.py --resume          # skip already-ok notebooks
    python run_pipeline_all.py --timeout 900     # per-notebook timeout
    python run_pipeline_all.py --family NLP      # single family
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import papermill as pm

ROOT = Path(__file__).parent
RESULTS_FILE = ROOT / "_notebook_results.json"
EXCLUDE_DIRS = {"venv", ".venv", "core", "data", "__pycache__", ".git", ".github"}

# Timeout overrides per family (seconds)
FAMILY_TIMEOUTS = {
    "anomaly": 600,
    "fraud": 600,
    "deep learning": 900,
    "computer vision": 900,
    "nlp": 900,
    "speech": 900,
    "reinforcement": 900,
    "time series": 600,
}


def discover_notebooks(family_filter=""):
    """Find all notebooks adjacent to pipeline.py."""
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
            if not family_filter or family_filter.lower() in str(nb_path).lower():
                nbs.append(nb_path)
    return nbs


def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text("utf-8"))
    return {}


def save_results(results):
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), "utf-8")


def get_timeout(nb_path: Path, default: int) -> int:
    """Look up per-family timeout override."""
    path_lower = str(nb_path).lower()
    for key, t in FAMILY_TIMEOUTS.items():
        if key in path_lower:
            return t
    return default


def run_one(nb_path: Path, timeout: int) -> dict:
    t0 = time.perf_counter()
    timeout = get_timeout(nb_path, timeout)
    try:
        pm.execute_notebook(
            str(nb_path),
            str(nb_path),
            cwd=str(nb_path.parent),
            kernel_name="python3",
            timeout=timeout,
            progress_bar=False,
        )
        return {"status": "ok", "time_s": round(time.perf_counter() - t0, 1), "error": None}
    except pm.PapermillExecutionError as exc:
        return {
            "status": "error",
            "time_s": round(time.perf_counter() - t0, 1),
            "error": f"Cell {exc.cell_index}: {exc.ename}: {exc.evalue}",
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 1)
        exc_name = type(exc).__name__
        if "Timeout" in exc_name or "timeout" in str(exc).lower():
            return {"status": "timeout", "time_s": elapsed, "error": f"Timeout after {timeout}s"}
        return {"status": "error", "time_s": elapsed, "error": f"{exc_name}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--family", type=str, default="")
    parser.add_argument("--start", type=int, default=0, help="0-based notebook start index")
    args = parser.parse_args()

    nbs = discover_notebooks(args.family)
    print(f"Discovered {len(nbs)} pipeline notebooks")
    if args.family:
        print(f"Filtered to family: {args.family}")
    if args.start > 0:
        nbs = nbs[args.start:]
        print(f"Starting from index {args.start}, {len(nbs)} remaining")

    results = load_results() if args.resume else {}
    ok = fail = skip = 0

    for i, nb_path in enumerate(nbs, 1):
        key = nb_path.relative_to(ROOT).as_posix()
        if args.resume and results.get(key, {}).get("status") == "ok":
            skip += 1
            continue

        family = nb_path.parent.parent.name
        print(f"\n[{i}/{len(nbs)}] [{family}] {nb_path.parent.name}")
        
        result = run_one(nb_path, args.timeout)
        results[key] = result
        
        if result["status"] == "ok":
            ok += 1
            print(f"  ✓ OK ({result['time_s']}s)")
        else:
            fail += 1
            err = (result.get("error") or "")[:120]
            print(f"  ✗ FAIL ({result['time_s']}s): {err}")
        
        save_results(results)

    print(f"\n{'='*65}")
    print(f"PIPELINE RESULTS: {ok} OK  |  {fail} FAIL  |  {skip} SKIPPED")
    print(f"{'='*65}")

    if fail:
        print("\nFailed notebooks:")
        for key, res in sorted(results.items()):
            if res.get("status") != "ok":
                print(f"  [{res['status']}] {key.split('/')[-1]}: {(res.get('error') or '')[:80]}")
    
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
