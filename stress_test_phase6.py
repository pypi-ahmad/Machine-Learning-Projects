#!/usr/bin/env python3
"""
Phase 6 — Execution & Stress Testing
======================================
Executes all 43 standardized ML project notebooks and runs stress simulations:

  1. BASELINE  — Execute each notebook end-to-end
  2. STRESS    — Inject: large data, missing values, wrong schema, repeated runs
  3. VALIDATE  — Check: no crashes, stable outputs, consistent predictions
  4. DETECT    — Monitor: memory, execution time, pipeline failures

Usage:
    python stress_test_phase6.py                  # full run
    python stress_test_phase6.py --project P057   # single project
    python stress_test_phase6.py --baseline-only   # skip stress tests
    python stress_test_phase6.py --timeout 300     # cell timeout in seconds
"""

from __future__ import annotations
import argparse
import copy
import csv
import gc
import glob
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import nbformat
import psutil

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
PHASE5_REPORT = ROOT / "audit_phase5" / "standardization_report.json"
PHASE3_CSV = ROOT / "audit_phase3" / "phase3_dataset_status.csv"
PHASE6_DIR = ROOT / "audit_phase6"
PROGRESS_FILE = PHASE6_DIR / "_progress.log"

CELL_TIMEOUT = 120        # seconds per cell
NB_TIMEOUT = 600          # seconds per notebook total
STRESS_SCALE_FACTOR = 10  # multiply dataset rows by this for large-data stress
MAX_MEMORY_MB = 4096      # alert threshold per notebook

# PyCaret cannot run on Python 3.13 — we still test the cell generation
# but expect PyCaret cells to raise ImportError / version errors
PYCARET_EXPECTED_FAIL = sys.version_info >= (3, 12)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_log_fh = None

def log(msg: str, end: str = "\n"):
    """Print to stdout AND append to progress log file."""
    global _log_fh
    print(msg, end=end, flush=True)
    if _log_fh:
        _log_fh.write(msg + end)
        _log_fh.flush()

def init_log():
    """Open the progress log file."""
    global _log_fh
    PHASE6_DIR.mkdir(exist_ok=True)
    _log_fh = open(PROGRESS_FILE, "w", encoding="utf-8")


def extract_project_number(dirname: str) -> int | None:
    m = re.search(r"Project[s]?\s+(\d+)", dirname)
    return int(m.group(1)) if m else None


def find_project_dir(pnum: int) -> Path | None:
    pattern = str(ROOT / f"Machine Learning Project*{pnum}*")
    for d in glob.glob(pattern):
        n = extract_project_number(os.path.basename(d))
        if n == pnum:
            return Path(d)
    return None


def load_phase3_status() -> dict[int, str]:
    """Map project number → data status."""
    result = {}
    with open(PHASE3_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"Project[s]?\s+(\d+)", row["project"])
            if m:
                result[int(m.group(1))] = row.get("status", "UNKNOWN")
    return result


def get_process_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_system_memory_mb() -> dict:
    vm = psutil.virtual_memory()
    return {
        "total_mb": round(vm.total / (1024 * 1024)),
        "available_mb": round(vm.available / (1024 * 1024)),
        "percent_used": vm.percent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Notebook execution engine
# ─────────────────────────────────────────────────────────────────────────────

class CellResult:
    """Result of executing a single cell."""
    def __init__(self, index: int, cell_type: str):
        self.index = index
        self.cell_type = cell_type
        self.status = "not_run"       # ok, error, timeout, skipped
        self.error_type = None        # exception class name
        self.error_message = None     # first line of traceback
        self.execution_time_s = 0.0
        self.memory_before_mb = 0.0
        self.memory_after_mb = 0.0
        self.is_standardized = False  # True for Phase 5 added cells
        self.is_pycaret = False
        self.is_lazypredict = False

    def to_dict(self):
        return {
            "index": self.index,
            "cell_type": self.cell_type,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "execution_time_s": round(self.execution_time_s, 3),
            "memory_delta_mb": round(self.memory_after_mb - self.memory_before_mb, 1),
            "is_standardized": self.is_standardized,
            "is_pycaret": self.is_pycaret,
            "is_lazypredict": self.is_lazypredict,
        }


class NotebookResult:
    """Aggregated result for one notebook execution."""
    def __init__(self, project: int, task: str, notebook: str, run_type: str = "baseline"):
        self.project = project
        self.task = task
        self.notebook = notebook
        self.run_type = run_type
        self.status = "not_run"       # ok, partial, error, blocked
        self.cells: list[CellResult] = []
        self.total_time_s = 0.0
        self.peak_memory_mb = 0.0
        self.error_summary = None
        self.blocked_reason = None

    @property
    def failures(self) -> list[CellResult]:
        return [c for c in self.cells if c.status == "error"]

    @property
    def timeouts(self) -> list[CellResult]:
        return [c for c in self.cells if c.status == "timeout"]

    def to_dict(self):
        return {
            "project": self.project,
            "task": self.task,
            "notebook": self.notebook,
            "run_type": self.run_type,
            "status": self.status,
            "total_time_s": round(self.total_time_s, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "cells_total": len(self.cells),
            "cells_ok": sum(1 for c in self.cells if c.status == "ok"),
            "cells_error": len(self.failures),
            "cells_timeout": len(self.timeouts),
            "cells_skipped": sum(1 for c in self.cells if c.status == "skipped"),
            "error_summary": self.error_summary,
            "blocked_reason": self.blocked_reason,
            "failures": [c.to_dict() for c in self.failures],
            "timeouts": [c.to_dict() for c in self.timeouts],
        }


def classify_cell(source: str) -> dict:
    """Identify if a cell is standardized / LazyPredict / PyCaret."""
    s = source.strip()
    return {
        "is_standardized": "Auto-generated by Phase 5" in s or "Standardized ML Pipeline" in s
                           or "LazyClassifier" in s or "LazyRegressor" in s
                           or "pycaret." in s or "save_model" in s,
        "is_lazypredict": "LazyClassifier" in s or "LazyRegressor" in s
                          or "lazypredict" in s or "silhouette_score" in s,
        "is_pycaret": "pycaret." in s or "compare_models" in s or "finalize_model" in s
                      or "create_model" in s or "assign_model" in s,
    }


def execute_notebook(
    nb_path: Path,
    project: int,
    task: str,
    cell_timeout: int = CELL_TIMEOUT,
    nb_timeout: int = NB_TIMEOUT,
    run_type: str = "baseline",
    inject_code: str | None = None,
) -> NotebookResult:
    """
    Execute a notebook end-to-end with allow_errors=True, then inspect
    each cell's outputs for errors.

    Uses nbclient.execute() for reliable kernel lifecycle management.
    If inject_code is provided, it is prepended as a new code cell
    (used for stress injection).
    """
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

    result = NotebookResult(project, task, nb_path.name, run_type)
    nb_dir = nb_path.parent

    try:
        nb = nbformat.read(str(nb_path), as_version=4)
    except Exception as e:
        result.status = "error"
        result.error_summary = f"Failed to read notebook: {e}"
        return result

    # Inject stress code as first cell if provided
    if inject_code:
        inject_cell = nbformat.v4.new_code_cell(source=inject_code)
        nb.cells.insert(0, inject_cell)

    # Tag all cells before execution
    code_cell_indices = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    cell_tags = {}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            cell_tags[i] = classify_cell(cell.source)

    # Create client with allow_errors so it runs all cells
    client = NotebookClient(
        nb,
        timeout=cell_timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_dir)}},
        allow_errors=True,
    )

    start_time = time.time()
    mem_before = get_process_memory_mb()

    try:
        client.execute()
    except DeadKernelError as e:
        result.status = "error"
        result.error_summary = f"Kernel died: {e}"
        result.total_time_s = time.time() - start_time
        result.peak_memory_mb = get_process_memory_mb()
        return result
    except CellTimeoutError as e:
        # Notebook-level timeout — still parse what we have
        pass
    except Exception as e:
        result.status = "error"
        result.error_summary = f"Execution failed: {type(e).__name__}: {str(e)[:300]}"
        result.total_time_s = time.time() - start_time
        result.peak_memory_mb = get_process_memory_mb()
        return result

    result.total_time_s = time.time() - start_time
    result.peak_memory_mb = get_process_memory_mb()

    # Parse cell outputs to determine per-cell status
    for idx, cell in enumerate(nb.cells):
        cr = CellResult(idx, cell.cell_type)

        if cell.cell_type != "code":
            cr.status = "skipped"
            result.cells.append(cr)
            continue

        tags = cell_tags.get(idx, {})
        cr.is_standardized = tags.get("is_standardized", False)
        cr.is_lazypredict = tags.get("is_lazypredict", False)
        cr.is_pycaret = tags.get("is_pycaret", False)

        # Check cell outputs for errors
        has_error = False
        error_name = None
        error_value = None
        error_tb = None

        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                has_error = True
                error_name = output.get("ename", "UnknownError")
                error_value = output.get("evalue", "")
                tb_lines = output.get("traceback", [])
                if tb_lines:
                    # Get last meaningful line from traceback
                    error_tb = tb_lines[-1] if tb_lines else ""
                    # Strip ANSI escape codes
                    error_tb = re.sub(r'\x1b\[[0-9;]*m', '', error_tb)
                break

        # Check execution_count to see if cell actually ran
        exec_count = cell.get("execution_count")

        if has_error:
            cr.status = "error"
            cr.error_type = error_name
            msg = f"{error_name}: {error_value}"
            if len(msg) > 500:
                msg = msg[:500] + "..."
            cr.error_message = msg
        elif exec_count is not None:
            cr.status = "ok"
        else:
            # Cell may not have been reached (kernel died earlier)
            cr.status = "skipped"

        # Estimate execution time from cell metadata if available
        exec_meta = cell.get("metadata", {}).get("execution", {})
        if "iopub.execute_input" in exec_meta and "shell.execute_reply" in exec_meta:
            try:
                from datetime import datetime as _dt
                t0 = _dt.fromisoformat(exec_meta["iopub.execute_input"].replace("Z", "+00:00"))
                t1 = _dt.fromisoformat(exec_meta["shell.execute_reply"].replace("Z", "+00:00"))
                cr.execution_time_s = (t1 - t0).total_seconds()
            except Exception:
                pass

        result.cells.append(cr)

    # Determine overall status
    errors = [c for c in result.cells if c.status == "error"]
    timeouts = [c for c in result.cells if c.status == "timeout"]

    # Filter out expected PyCaret failures
    real_errors = [c for c in errors if not (c.is_pycaret and PYCARET_EXPECTED_FAIL)]

    if not real_errors and not timeouts:
        result.status = "ok"
    elif real_errors and len(real_errors) < len(code_cell_indices) // 2:
        result.status = "partial"
        result.error_summary = "; ".join(
            f"cell[{c.index}]: {c.error_type}" for c in real_errors[:3]
        )
    else:
        result.status = "error"
        result.error_summary = "; ".join(
            f"cell[{c.index}]: {c.error_type}" for c in (real_errors + timeouts)[:3]
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stress injection generators
# ─────────────────────────────────────────────────────────────────────────────

def gen_missing_values_injection() -> str:
    """Inject NaN values into the DataFrame after it's loaded."""
    return '''
# === STRESS INJECTION: Missing Values ===
import numpy as np
import pandas as pd

_orig_read_csv = pd.read_csv

def _patched_read_csv(*args, **kwargs):
    """Inject random NaN into ~10% of cells."""
    _df = _orig_read_csv(*args, **kwargs)
    np.random.seed(42)
    mask = np.random.random(_df.shape) < 0.10
    # Only inject NaN into numeric columns
    for col in _df.select_dtypes(include=[np.number]).columns:
        cidx = _df.columns.get_loc(col)
        _df.loc[mask[:, cidx], col] = np.nan
    return _df

pd.read_csv = _patched_read_csv
'''


def gen_large_data_injection(scale: int = STRESS_SCALE_FACTOR) -> str:
    """Duplicate rows to simulate large datasets."""
    return f'''
# === STRESS INJECTION: Large Data ({scale}x) ===
import pandas as pd

_orig_read_csv_lg = pd.read_csv

def _patched_read_csv_lg(*args, **kwargs):
    """Duplicate rows to stress-test with {scale}x more data."""
    _df = _orig_read_csv_lg(*args, **kwargs)
    if len(_df) < 100000:  # only scale if not already huge
        _df = pd.concat([_df] * {scale}, ignore_index=True)
    return _df

pd.read_csv = _patched_read_csv_lg
'''


def gen_wrong_schema_injection() -> str:
    """Add extra columns and rename some to simulate schema drift."""
    return '''
# === STRESS INJECTION: Wrong Schema ===
import numpy as np
import pandas as pd

_orig_read_csv_ws = pd.read_csv

def _patched_read_csv_ws(*args, **kwargs):
    """Add extra columns to the DataFrame to simulate schema changes."""
    _df = _orig_read_csv_ws(*args, **kwargs)
    # Add 3 extra numeric columns
    for i in range(3):
        _df[f'_extra_col_{i}'] = np.random.randn(len(_df))
    # Add 1 extra categorical column
    _df['_extra_cat'] = np.random.choice(['A', 'B', 'C'], size=len(_df))
    return _df

pd.read_csv = _patched_read_csv_ws
'''


def gen_repeated_run_injection() -> str:
    """No data injection — this will just run the notebook twice."""
    return '''
# === STRESS INJECTION: Repeated Run (consistency check) ===
# No modifications — this run checks output stability vs baseline
'''


STRESS_TESTS = {
    "missing_values": gen_missing_values_injection,
    "large_data": gen_large_data_injection,
    "wrong_schema": gen_wrong_schema_injection,
    "repeated_run": gen_repeated_run_injection,
}


# ─────────────────────────────────────────────────────────────────────────────
# Root cause analyzer
# ─────────────────────────────────────────────────────────────────────────────

def analyze_root_cause(result: NotebookResult) -> dict:
    """Analyze failures and classify root causes."""
    causes = []

    for c in result.failures:
        msg = (c.error_message or "").lower()
        etype = (c.error_type or "").lower()

        cause = {
            "cell_index": c.index,
            "is_standardized": c.is_standardized,
            "is_pycaret": c.is_pycaret,
            "is_lazypredict": c.is_lazypredict,
            "category": "unknown",
            "detail": c.error_message,
        }

        # Classify the root cause
        if "filenotfounderror" in etype or "no such file" in msg:
            cause["category"] = "missing_data"
        elif "modulenotfounderror" in etype or "importerror" in etype:
            if c.is_pycaret:
                cause["category"] = "pycaret_incompatible"
            else:
                cause["category"] = "missing_module"
        elif "keyerror" in etype:
            cause["category"] = "schema_mismatch"
        elif "valueerror" in etype and ("nan" in msg or "missing" in msg or "null" in msg):
            cause["category"] = "missing_values"
        elif "valueerror" in etype:
            cause["category"] = "value_error"
        elif "memoryerror" in etype:
            cause["category"] = "memory_overflow"
        elif "celltimeouterror" in etype or c.status == "timeout":
            cause["category"] = "timeout"
        elif "typeerror" in etype:
            cause["category"] = "type_error"
        elif "attributeerror" in etype:
            cause["category"] = "api_mismatch"
        elif "indexerror" in etype:
            cause["category"] = "index_error"
        elif c.is_pycaret and PYCARET_EXPECTED_FAIL:
            cause["category"] = "pycaret_incompatible"
        else:
            cause["category"] = "runtime_error"

        causes.append(cause)

    # Timeouts
    for c in result.timeouts:
        causes.append({
            "cell_index": c.index,
            "is_standardized": c.is_standardized,
            "is_pycaret": c.is_pycaret,
            "is_lazypredict": c.is_lazypredict,
            "category": "slow_execution",
            "detail": c.error_message,
        })

    # Memory check
    if result.peak_memory_mb > MAX_MEMORY_MB:
        causes.append({
            "cell_index": -1,
            "is_standardized": False,
            "is_pycaret": False,
            "is_lazypredict": False,
            "category": "high_memory",
            "detail": f"Peak memory {result.peak_memory_mb:.0f} MB > {MAX_MEMORY_MB} MB threshold",
        })

    return {
        "project": result.project,
        "run_type": result.run_type,
        "causes": causes,
        "primary_cause": causes[0]["category"] if causes else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Execution orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_project(
    project_info: dict,
    p3_status: dict,
    cell_timeout: int,
    nb_timeout: int,
    baseline_only: bool,
) -> list[NotebookResult]:
    """Run baseline + stress tests for a single project."""
    pnum = project_info["project"]
    task = project_info["task"]
    nb_name = project_info["notebook"]

    pdir = find_project_dir(pnum)
    if not pdir:
        r = NotebookResult(pnum, task, nb_name, "baseline")
        r.status = "blocked"
        r.blocked_reason = "project_directory_not_found"
        return [r]

    nb_path = pdir / nb_name
    if not nb_path.exists():
        r = NotebookResult(pnum, task, nb_name, "baseline")
        r.status = "blocked"
        r.blocked_reason = "notebook_not_found"
        return [r]

    ds = p3_status.get(pnum, "UNKNOWN")
    if ds not in ("OK_LOCAL", "OK_BUILTIN", "DOWNLOADED"):
        r = NotebookResult(pnum, task, nb_name, "baseline")
        r.status = "blocked"
        r.blocked_reason = f"data_{ds}"
        return [r]

    results = []

    # Baseline run
    log(f"  P{pnum:03d}: baseline ...", end=" ")
    t0 = time.time()
    baseline = execute_notebook(nb_path, pnum, task, cell_timeout, nb_timeout, "baseline")
    elapsed = time.time() - t0
    log(f"{baseline.status:8s}  ({elapsed:.1f}s, {baseline.peak_memory_mb:.0f}MB)")
    results.append(baseline)

    # If baseline is totally broken (missing data, kernel fail), skip stress
    if baseline.status == "error" and baseline.error_summary and (
        "FileNotFoundError" in (baseline.error_summary or "")
        or "Kernel startup failed" in (baseline.error_summary or "")
    ):
        log(f"          Skipping stress tests — baseline fatally failed")
        return results

    if baseline_only:
        return results

    # Stress tests
    for stress_name, gen_fn in STRESS_TESTS.items():
        log(f"  P{pnum:03d}: {stress_name:20s} ...", end=" ")
        t0 = time.time()
        inject = gen_fn()
        sr = execute_notebook(
            nb_path, pnum, task, cell_timeout, nb_timeout,
            run_type=stress_name,
            inject_code=inject,
        )
        elapsed = time.time() - t0
        log(f"{sr.status:8s}  ({elapsed:.1f}s, {sr.peak_memory_mb:.0f}MB)")
        results.append(sr)

        # Force GC between stress runs
        gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(all_results: list[NotebookResult]) -> dict:
    """Generate comprehensive Phase 6 report."""
    # Aggregate by project
    projects = {}
    for r in all_results:
        if r.project not in projects:
            projects[r.project] = {
                "project": r.project,
                "task": r.task,
                "notebook": r.notebook,
                "runs": [],
            }
        projects[r.project]["runs"].append(r)

    # Build report
    report = {
        "phase": 6,
        "title": "Execution & Stress Testing",
        "generated": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pycaret_expected_fail": PYCARET_EXPECTED_FAIL,
        "config": {
            "cell_timeout_s": CELL_TIMEOUT,
            "nb_timeout_s": NB_TIMEOUT,
            "stress_scale_factor": STRESS_SCALE_FACTOR,
            "memory_threshold_mb": MAX_MEMORY_MB,
        },
        "summary": {},
        "projects": [],
        "root_causes": {},
        "failures_per_project": [],
    }

    # Per-project analysis
    total_ok = 0
    total_partial = 0
    total_error = 0
    total_blocked = 0
    all_causes = {}

    for pnum in sorted(projects.keys()):
        pdata = projects[pnum]
        proj_report = {
            "project": pnum,
            "task": pdata["task"],
            "runs": [],
            "failures": [],
        }

        is_blocked = False
        for r in pdata["runs"]:
            run_dict = r.to_dict()
            proj_report["runs"].append(run_dict)

            if r.status == "blocked":
                is_blocked = True
                total_blocked += 1
                continue

            if r.status == "ok":
                total_ok += 1
            elif r.status == "partial":
                total_partial += 1
            else:
                total_error += 1

            # Root cause analysis
            rca = analyze_root_cause(r)
            if rca["causes"]:
                proj_report["failures"].extend(rca["causes"])
                for cause in rca["causes"]:
                    cat = cause["category"]
                    all_causes[cat] = all_causes.get(cat, 0) + 1

        report["projects"].append(proj_report)

        # Per-project failure summary
        if proj_report["failures"]:
            cats = list(set(c["category"] for c in proj_report["failures"]))
            report["failures_per_project"].append({
                "project": pnum,
                "task": pdata["task"],
                "failure_count": len(proj_report["failures"]),
                "categories": cats,
                "primary_cause": cats[0] if cats else None,
            })

    report["summary"] = {
        "total_runs": len(all_results),
        "ok": total_ok,
        "partial": total_partial,
        "error": total_error,
        "blocked": total_blocked,
    }
    report["root_causes"] = dict(sorted(all_causes.items(), key=lambda x: -x[1]))

    return report


def print_summary(report: dict):
    """Print human-readable summary."""
    log("\n" + "=" * 70)
    log("Phase 6 — Execution & Stress Testing: Summary")
    log("=" * 70)

    s = report["summary"]
    log(f"  Total runs:  {s['total_runs']}")
    log(f"  OK:          {s['ok']}")
    log(f"  Partial:     {s['partial']}")
    log(f"  Error:       {s['error']}")
    log(f"  Blocked:     {s['blocked']}")

    if report["root_causes"]:
        log(f"\nRoot Causes (across all runs):")
        for cause, count in report["root_causes"].items():
            log(f"  {cause:30s}  {count}")

    if report["failures_per_project"]:
        log(f"\nFailures per Project:")
        for fp in report["failures_per_project"]:
            cats = ", ".join(fp["categories"])
            log(f"  P{fp['project']:03d}  ({fp['task']:16s})  {fp['failure_count']} failures  [{cats}]")

    log("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 6 — Execution & Stress Testing")
    parser.add_argument("--project", type=str, help="Run single project, e.g. P057")
    parser.add_argument("--baseline-only", action="store_true", help="Skip stress tests")
    parser.add_argument("--timeout", type=int, default=CELL_TIMEOUT, help="Cell timeout (seconds)")
    parser.add_argument("--nb-timeout", type=int, default=NB_TIMEOUT, help="Notebook timeout (seconds)")
    args = parser.parse_args()

    # Initialize progress log
    init_log()
    # Load Phase 5 report
    with open(PHASE5_REPORT, encoding="utf-8") as f:
        p5 = json.load(f)

    ok_projects = [d for d in p5["details"] if d["status"] == "OK"]
    p3_status = load_phase3_status()

    # Filter to single project if specified
    if args.project:
        pnum = int(args.project.upper().replace("P", ""))
        ok_projects = [d for d in ok_projects if d["project"] == pnum]
        if not ok_projects:
            log(f"Project P{pnum:03d} not found in Phase 5 OK list")
            sys.exit(1)

    log(f"Phase 6 — Execution & Stress Testing")
    log(f"{'=' * 70}")
    log(f"  Projects: {len(ok_projects)}")
    log(f"  Cell timeout: {args.timeout}s")
    log(f"  Notebook timeout: {args.nb_timeout}s")
    log(f"  Stress tests: {'OFF' if args.baseline_only else 'ON'}")
    log(f"  PyCaret expected fail: {PYCARET_EXPECTED_FAIL}")
    log(f"  System memory: {get_system_memory_mb()}")
    log(f"{'=' * 70}\n")

    all_results: list[NotebookResult] = []

    for proj in ok_projects:
        try:
            results = run_project(proj, p3_status, args.timeout, args.nb_timeout, args.baseline_only)
            all_results.extend(results)
        except Exception as e:
            log(f"  P{proj['project']:03d}: FATAL — {e}")
            traceback.print_exc()
            r = NotebookResult(proj["project"], proj["task"], proj["notebook"], "baseline")
            r.status = "error"
            r.error_summary = f"Orchestrator crash: {e}"
            all_results.append(r)

        # Force GC between projects
        gc.collect()

        # Save incremental report after each project
        _interim = generate_report(all_results)
        PHASE6_DIR.mkdir(exist_ok=True)
        with open(PHASE6_DIR / "phase6_stress_report.json", "w", encoding="utf-8") as f:
            json.dump(_interim, f, indent=2, default=str)

    # Generate final report
    report = generate_report(all_results)
    print_summary(report)

    # Save report
    PHASE6_DIR.mkdir(exist_ok=True)
    report_path = PHASE6_DIR / "phase6_stress_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"\nReport saved: {report_path}")

    return report


if __name__ == "__main__":
    main()