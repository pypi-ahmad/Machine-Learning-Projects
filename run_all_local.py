#!/usr/bin/env python3
"""
run_all_local.py — Execute all 319 pipeline.py files locally with GPU/CUDA.

Features:
  - Runs each pipeline.py in its own subprocess (isolated)
  - Incremental: resumes from last run (skips already-succeeded)
  - Classifies errors for batch fixing
  - Updates .ipynb notebooks with execution logs
  - Produces colab_execution_results.json with full report

Usage:
  python run_all_local.py                  # run all
  python run_all_local.py --family Clustering  # run one family
  python run_all_local.py --retry           # re-run only failures
  python run_all_local.py --timeout 600     # custom timeout (seconds)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
RESULTS_FILE = ROOT / "execution_results.json"
EXCLUDE = {"venv", ".venv", "core", "data", "__pycache__", ".git", ".github"}

# Always use the venv python, not system python
VENV_PYTHON = ROOT / "venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)  # fallback


def find_pipeline_dirs():
    dirs = []
    for p in sorted(ROOT.rglob("pipeline.py")):
        parts_lower = {x.lower() for x in p.parts}
        if parts_lower & EXCLUDE:
            continue
        if any("data analysis" in x.lower() for x in p.parts):
            continue
        dirs.append(p.parent)
    return dirs


def classify_error(stderr: str) -> tuple[str, str]:
    """Return (error_type, detail) from stderr."""
    if "ModuleNotFoundError" in stderr:
        m = re.search(r"No module named '(\S+)'", stderr)
        return "missing_module", m.group(1) if m else ""
    if "DatasetNotFoundError" in stderr:
        return "dataset_not_found", ""
    if "ConnectionError" in stderr or "URLError" in stderr or "HTTPError" in stderr:
        return "network", ""
    if "TypeError" in stderr:
        return "type_error", _last_err(stderr)
    if "ValueError" in stderr:
        return "value_error", _last_err(stderr)
    if "MemoryError" in stderr or "OOM" in stderr:
        return "oom", ""
    if "RuntimeError" in stderr:
        return "runtime_error", _last_err(stderr)
    if "FileNotFoundError" in stderr:
        return "file_not_found", _last_err(stderr)
    if "KeyError" in stderr:
        return "key_error", _last_err(stderr)
    return "other", _last_err(stderr)


def _last_err(stderr: str) -> str:
    lines = [
        l.strip()
        for l in stderr.splitlines()
        if l.strip()
        and not l.strip().startswith("File ")
        and not l.strip().startswith("Traceback")
        and not l.strip().startswith("During handling")
    ]
    return lines[-1][:200] if lines else ""


def run_pipeline(project_dir: Path, timeout: int = 300) -> dict:
    pipeline_path = project_dir / "pipeline.py"
    result = {
        "project": project_dir.name,
        "family": project_dir.parent.name,
        "path": str(project_dir),
        "status": "unknown",
        "time_s": 0,
        "artifacts": [],
        "error_type": "",
        "error_detail": "",
    }
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(VENV_PYTHON), str(pipeline_path)],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        elapsed = time.time() - t0
        result["time_s"] = round(elapsed, 1)
        result["status"] = "success" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            et, ed = classify_error(proc.stderr or "")
            result["error_type"] = et
            result["error_detail"] = ed
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["time_s"] = timeout
        result["error_type"] = "timeout"
    except Exception as e:
        result["status"] = "error"
        result["error_type"] = type(e).__name__
        result["error_detail"] = str(e)[:200]
        result["time_s"] = round(time.time() - t0, 1)

    # Collect artifacts produced
    for f in project_dir.iterdir():
        if f.is_file() and f.suffix in (
            ".json", ".csv", ".png", ".pkl", ".pt", ".onnx", ".jpg",
        ):
            result["artifacts"].append(f.name)
    return result


def update_notebook(project_dir: Path, run_result: dict):
    """Append execution log cell to the generated notebook."""
    try:
        import nbformat
    except ImportError:
        return
    safe_name = re.sub(r'[<>:"|?*]', "_", project_dir.name)
    nb_path = project_dir / f"{safe_name}.ipynb"
    if not nb_path.exists():
        return
    nb = nbformat.read(str(nb_path), as_version=4)
    from datetime import datetime

    log = (
        f"# Execution Log (Local GPU - {datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
        f"# Status: {run_result['status']}\n"
        f"# Time: {run_result['time_s']}s\n"
        f"# Artifacts: {', '.join(run_result['artifacts']) or 'none'}\n"
    )
    if run_result["error_type"]:
        log += f"# Error: {run_result['error_type']}: {run_result['error_detail']}\n"

    cell = nbformat.v4.new_code_cell(log)
    cell["metadata"] = {"tags": ["execution-log"]}

    # Replace existing log cell or append
    if nb.cells and "execution-log" in nb.cells[-1].get("metadata", {}).get("tags", []):
        nb.cells[-1] = cell
    else:
        nb.cells.append(cell)
    nbformat.write(nb, str(nb_path))


def load_existing() -> dict:
    if RESULTS_FILE.exists():
        try:
            data = json.loads(RESULTS_FILE.read_text("utf-8"))
            return {r["path"]: r for r in data}
        except Exception:
            pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run all ML pipelines locally")
    parser.add_argument("--family", help="Run only this family")
    parser.add_argument("--retry", action="store_true", help="Re-run only failures")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per pipeline (seconds)")
    parser.add_argument("--fresh", action="store_true", help="Ignore cached results")
    args = parser.parse_args()

    pipeline_dirs = find_pipeline_dirs()
    print(f"Found {len(pipeline_dirs)} pipelines")

    if args.family:
        pipeline_dirs = [d for d in pipeline_dirs if d.parent.name == args.family]
        print(f"Filtered to {len(pipeline_dirs)} in family '{args.family}'")

    existing = {} if args.fresh else load_existing()

    all_results = []
    skipped = 0
    for i, d in enumerate(pipeline_dirs, 1):
        key = str(d)

        # Skip logic
        if key in existing:
            prev = existing[key]
            if prev["status"] == "success" and not args.retry:
                all_results.append(prev)
                skipped += 1
                continue
            if not args.retry and prev["status"] != "success":
                pass  # re-run failures by default
            elif args.retry and prev["status"] == "success":
                all_results.append(prev)
                skipped += 1
                continue

        print(
            f"[{i}/{len(pipeline_dirs)}] {d.parent.name}/{d.name}...",
            end=" ",
            flush=True,
        )
        r = run_pipeline(d, timeout=args.timeout)
        all_results.append(r)

        icon = "OK" if r["status"] == "success" else "FAIL"
        arts = len(r["artifacts"])
        err = f" [{r['error_type']}]" if r["error_type"] else ""
        print(f"{icon} ({r['time_s']}s, {arts} artifacts){err}", flush=True)

        # Update notebook
        update_notebook(d, r)

        # Save incremental progress
        RESULTS_FILE.write_text(
            json.dumps(all_results, indent=1, ensure_ascii=False), "utf-8"
        )

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"EXECUTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total: {len(all_results)} | Skipped (cached): {skipped}")

    status_counts = Counter(r["status"] for r in all_results)
    for s, c in status_counts.most_common():
        print(f"  {s}: {c}")

    error_counts = Counter(
        r["error_type"] for r in all_results if r["status"] != "success" and r["error_type"]
    )
    if error_counts:
        print(f"\nError breakdown:")
        for e, c in error_counts.most_common():
            print(f"  {e}: {c}")

    # Missing modules summary
    missing = Counter(
        r["error_detail"]
        for r in all_results
        if r["error_type"] == "missing_module" and r["error_detail"]
    )
    if missing:
        print(f"\nMissing modules:")
        for m, c in missing.most_common():
            print(f"  pip install {m}  ({c} pipelines)")

    artifact_total = sum(len(r["artifacts"]) for r in all_results)
    print(f"\nTotal artifacts produced: {artifact_total}")
    print(f"Results saved to: {RESULTS_FILE}")

    failed = sum(1 for r in all_results if r["status"] != "success")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
