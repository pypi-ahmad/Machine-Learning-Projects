"""Run all *_pipeline.ipynb notebooks locally using jupyter nbconvert.

Each notebook is executed in-place with GPU/CUDA.
Progress is saved after each notebook so execution can resume.

Usage::
    python dashboard/scripts/run_all_notebooks.py
    python dashboard/scripts/run_all_notebooks.py --resume   # skip already-executed
    python dashboard/scripts/run_all_notebooks.py --timeout 600
"""

from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS_FILE = REPO / "notebook_execution_results.json"


def find_notebooks() -> list[str]:
    os.chdir(REPO)
    nbs = sorted(glob.glob("**/*_pipeline.ipynb", recursive=True))
    return nbs


def load_results() -> list[dict]:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return []


def save_results(results: list[dict]):
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")


def run_notebook(nb_path: str, timeout: int) -> dict:
    start = time.time()
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                f"--ExecutePreprocessor.timeout={timeout}",
                "--ExecutePreprocessor.kernel_name=python3",
                "--inplace",
                nb_path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 60,  # extra margin
            cwd=str(REPO),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        elapsed = time.time() - start
        if proc.returncode == 0:
            return {"notebook": nb_path, "status": "success", "time": round(elapsed, 1), "error": ""}
        else:
            err = proc.stderr.strip().split("\n")[-1][:300] if proc.stderr else ""
            return {"notebook": nb_path, "status": "partial", "time": round(elapsed, 1), "error": err}
    except subprocess.TimeoutExpired:
        return {"notebook": nb_path, "status": "timeout", "time": round(time.time() - start, 1), "error": "timeout"}
    except Exception as e:
        return {"notebook": nb_path, "status": "failed", "time": round(time.time() - start, 1), "error": str(e)[:300]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip already-executed notebooks")
    parser.add_argument("--timeout", type=int, default=300, help="Per-notebook timeout (seconds)")
    args = parser.parse_args()

    notebooks = find_notebooks()
    total = len(notebooks)
    print(f"Found {total} pipeline notebooks")

    results = load_results() if args.resume else []
    done_set = {r["notebook"] for r in results}

    for i, nb_path in enumerate(notebooks, 1):
        if nb_path in done_set:
            print(f"  [{i:3d}/{total}] SKIP (already done): {os.path.basename(nb_path)}")
            continue

        print(f"\n  [{i:3d}/{total}] {nb_path}")
        r = run_notebook(nb_path, args.timeout)
        results.append(r)
        save_results(results)

        icon = {"success": "OK", "partial": "WARN", "timeout": "TIME", "failed": "FAIL"}[r["status"]]
        print(f"           {icon} {r['time']:.0f}s", end="")
        if r["error"]:
            print(f"  -- {r['error'][:80]}", end="")
        print()

    # Summary
    ok = sum(1 for r in results if r["status"] == "success")
    part = sum(1 for r in results if r["status"] == "partial")
    fail = sum(1 for r in results if r["status"] in ("failed", "timeout"))
    tot_time = sum(r["time"] for r in results)
    print(f"\n{'='*60}")
    print(f"  Success: {ok}/{total}  Partial: {part}/{total}  Failed: {fail}/{total}")
    print(f"  Total time: {tot_time:.0f}s ({tot_time/60:.1f}min)")
    print(f"  Results: {RESULTS_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
