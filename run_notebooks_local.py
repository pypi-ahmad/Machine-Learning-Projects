#!/usr/bin/env python3
"""
run_notebooks_local.py
======================
Execute every generated .ipynb notebook end-to-end locally using GPU/CUDA.

Uses subprocess isolation: each notebook runs via ``jupyter nbconvert
--execute`` in a completely separate process.  This avoids kernel hang
issues and gives a clean process-level timeout per notebook.

Resumable: previously successful runs (tracked in execution_results.json)
are skipped automatically.  Delete that file to force a full re-run.
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
RESULTS_FILE = ROOT / "execution_results.json"
EXCLUDE_DIRS = {"venv", ".venv", "core", "data", "__pycache__", ".git", ".github"}
NOTEBOOK_TIMEOUT = 1800   # total seconds per notebook
# ─────────────────────────────────────────────────────────────────────────


def find_generated_notebooks() -> list[Path]:
    """Return paths to all generated notebooks (those next to a pipeline.py)."""
    notebooks = []
    for pipeline in sorted(ROOT.rglob("pipeline.py")):
        parts_lower = {p.lower() for p in pipeline.parts}
        if parts_lower & EXCLUDE_DIRS:
            continue
        if any("data analysis" in p.lower() for p in pipeline.parts):
            continue
        project_dir = pipeline.parent
        safe_name = re.sub(r'[<>:"|?*]', "_", project_dir.name)
        nb_path = project_dir / f"{safe_name}.ipynb"
        if nb_path.exists():
            notebooks.append(nb_path)
    return notebooks


def load_results() -> dict:
    """Load cached execution results."""
    if RESULTS_FILE.exists():
        try:
            data = json.loads(RESULTS_FILE.read_text("utf-8"))
            return {r["notebook"]: r for r in data}
        except Exception:
            return {}
    return {}


def save_results(results: dict):
    """Persist execution results."""
    RESULTS_FILE.write_text(
        json.dumps(list(results.values()), indent=1, default=str),
        encoding="utf-8",
    )


def collect_artifacts(project_dir: Path) -> list[str]:
    """List generated artifact files in the project directory."""
    exts = {".json", ".csv", ".png", ".pkl", ".pt", ".onnx", ".jpg", ".jpeg"}
    arts = []
    for f in sorted(project_dir.iterdir()):
        if f.suffix.lower() in exts and f.name not in ("pipeline.py",):
            arts.append(f.name)
    return arts


def execute_notebook(nb_path: Path) -> dict:
    """Execute a single notebook via subprocess (jupyter nbconvert --execute).

    The notebook is executed in-place, with outputs saved back to the file.
    """
    project_dir = nb_path.parent
    project_name = project_dir.name
    family = project_dir.parent.name

    result = {
        "notebook": str(nb_path),
        "project": project_name,
        "family": family,
        "status": "unknown",
        "time_s": 0,
        "artifacts": [],
        "error": "",
    }

    t0 = time.time()
    hard_timeout = NOTEBOOK_TIMEOUT + 60  # extra margin for startup/cleanup
    try:
        # Use Popen + CREATE_NEW_PROCESS_GROUP so we can kill the entire
        # process tree (nbconvert + kernel) on timeout – subprocess.run()
        # on Windows only kills the direct child, leaving zombie kernels.
        proc = subprocess.Popen(
            [
                str(VENV_PYTHON), "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=" + str(NOTEBOOK_TIMEOUT),
                "--ExecutePreprocessor.kernel_name=python3",
                str(nb_path),
            ],
            cwd=str(project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "0"},
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        try:
            stdout, stderr = proc.communicate(timeout=hard_timeout)
        except subprocess.TimeoutExpired:
            # Kill entire process tree via taskkill /T (tree kill)
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
            proc.wait(timeout=10)
            raise

        result["time_s"] = round(time.time() - t0, 1)
        stderr = stderr or ""

        if proc.returncode == 0:
            result["status"] = "success"
        else:
            result["status"] = "cell_error"
            # Extract the actual error from stderr
            err_lines = [
                l.strip() for l in stderr.splitlines()
                if l.strip()
                and not l.strip().startswith("Traceback")
                and not l.strip().startswith("File ")
                and not l.strip().startswith("During handling")
                and "warnings.warn" not in l
                and "MissingIDFieldWarning" not in l
            ]
            # Get the last meaningful error line
            result["error"] = err_lines[-1][:300] if err_lines else f"exit code {proc.returncode}"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["time_s"] = NOTEBOOK_TIMEOUT
        result["error"] = f"Notebook timed out after {NOTEBOOK_TIMEOUT}s"
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        result["time_s"] = round(time.time() - t0, 1)

    result["artifacts"] = collect_artifacts(project_dir)
    return result


def main():
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    notebooks = find_generated_notebooks()
    print(f"Found {len(notebooks)} generated notebooks")
    print(f"Notebook timeout: {NOTEBOOK_TIMEOUT}s")
    print(f"Python: {VENV_PYTHON}")
    print(f"Results file: {RESULTS_FILE}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print()

    # Load cached results for resume
    results = load_results()
    cached = sum(1 for r in results.values() if r["status"] == "success")
    if cached:
        print(f"Resuming: {cached} previously successful runs will be skipped\n")

    success = fail = skip = 0

    for i, nb_path in enumerate(notebooks, 1):
        key = str(nb_path)
        project = nb_path.parent.name
        family = nb_path.parent.parent.name

        # Skip cached successes
        if key in results and results[key]["status"] == "success":
            skip += 1
            print(f"[{i}/{len(notebooks)}] SKIP {family}/{project}")
            continue

        print(f"[{i}/{len(notebooks)}] RUN  {family}/{project} ...", end=" ", flush=True)

        r = execute_notebook(nb_path)
        results[key] = r

        if r["status"] == "success":
            success += 1
            print(f"OK ({r['time_s']}s, {len(r['artifacts'])} artifacts)")
        else:
            fail += 1
            print(f"FAIL [{r['status']}] ({r['time_s']}s)")
            if r["error"]:
                print(f"        {r['error'][:120]}")

        # Save progress incrementally
        save_results(results)

    # ── Summary ──────────────────────────────────────────────────────
    all_results = list(results.values())
    print(f"\n{'='*70}")
    print(f"EXECUTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total notebooks : {len(notebooks)}")
    print(f"  Success         : {success} (new) + {skip} (cached)")
    print(f"  Failed          : {fail}")

    if fail:
        print(f"\nFailure breakdown:")
        statuses = Counter(
            r["status"] for r in all_results if r["status"] != "success"
        )
        for s, c in statuses.most_common():
            print(f"  {s}: {c}")

        # Show most common errors
        print(f"\nTop errors:")
        errors = Counter(
            r["error"].split(":")[0] if r["error"] else "unknown"
            for r in all_results
            if r["status"] != "success"
        )
        for e, c in errors.most_common(10):
            print(f"  ({c}x) {e}")

    art_total = sum(len(r["artifacts"]) for r in all_results)
    print(f"\nTotal artifacts: {art_total}")
    print(f"Results saved: {RESULTS_FILE}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
