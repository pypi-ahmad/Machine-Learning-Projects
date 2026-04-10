from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

REPO = Path(__file__).resolve().parents[1]
RESULTS_FILE = REPO / "notebook_execution_results.json"
DEFAULT_TIMEOUT = 180

def discover_notebooks():
    os.chdir(REPO)
    return sorted(glob.glob("**/*_pipeline.ipynb", recursive=True))

def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return []

def save_results(results):
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

def run_notebook(nb_path, cell_timeout):
    start = time.time()
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb, timeout=cell_timeout, kernel_name="python3",
        resources={"metadata": {"path": str(Path(nb_path).resolve().parent)}},
    )
    status, error_msg = "success", ""
    try:
        client.execute()
    except CellTimeoutError:
        status, error_msg = "partial", f"Cell timeout ({cell_timeout}s)"
    except CellExecutionError as e:
        status = "partial"
        lines = str(e).strip().split("\n")
        error_msg = lines[-1][:200] if lines else "CellExecutionError"
    except Exception as e:
        status, error_msg = "partial", f"{type(e).__name__}: {str(e)[:150]}"
    finally:
        try: nbformat.write(nb, nb_path)
        except Exception: pass
        try: client.cleanup_kernel()
        except Exception: pass
    elapsed = time.time() - start
    return {"notebook": nb_path, "status": status, "time": round(elapsed, 1), "error": error_msg}

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()
    notebooks = discover_notebooks()
    total = len(notebooks)
    print(f"Discovered {total} pipeline notebooks")
    print(f"Cell timeout: {args.timeout}s")
    print(f"Results: {RESULTS_FILE}\n", flush=True)
    if args.fresh and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
        print("Cleared previous results\n", flush=True)
    results = load_results()
    done_set = {r["notebook"] for r in results}
    if done_set:
        print(f"Resuming: {len(done_set)}/{total} already done\n", flush=True)
    for i, nb_path in enumerate(notebooks, 1):
        if nb_path in done_set:
            continue
        name = os.path.basename(nb_path).replace("_pipeline.ipynb", "")
        print(f"[{i:3d}/{total}] {name} ... ", end="", flush=True)
        result = run_notebook(nb_path, args.timeout)
        results.append(result)
        save_results(results)
        tag = "OK" if result["status"] == "success" else "WARN"
        print(f"{tag}  {result['time']:.0f}s", flush=True)
        if result["error"]:
            print(f"         {result['error'][:120]}", flush=True)
    ok = sum(1 for r in results if r["status"] == "success")
    warn = sum(1 for r in results if r["status"] == "partial")
    tot_time = sum(r["time"] for r in results)
    print(f"\n{'=' * 60}")
    print(f"  DONE: {ok} OK | {warn} WARN  out of {total}")
    print(f"  Total time: {tot_time:.0f}s ({tot_time / 60:.1f}min)")
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
