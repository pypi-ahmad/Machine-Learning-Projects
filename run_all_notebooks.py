"""
Parallel notebook batch runner.
Runs all never-executed notebooks using a worker pool, saves progress to run_results.json.
"""
import json, os, glob, sys, time, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

REPO    = Path(r"e:\Github\Machine-Learning-Projects")
KERNEL  = "ml-projects"
TIMEOUT = 600          # seconds per notebook
PY      = r"C:\Users\ahmad\AppData\Local\Programs\Python\Python313\python.exe"
WORKERS = 4            # parallel notebooks; reduce if memory-constrained
RESULTS_FILE = REPO / "run_results.json"

lock = Lock()
results: dict = {}

# ── helpers ──────────────────────────────────────────────────────────────────

def find_never_run() -> list[Path]:
    nbs = [Path(p) for p in glob.glob(str(REPO / "**/*.ipynb"), recursive=True)
           if "venv" not in p and "__pycache__" not in p]
    never = []
    for p in nbs:
        try:
            with open(p, encoding="utf-8") as f:
                nb = json.load(f)
            has_out = any(c.get("outputs")
                          for c in nb.get("cells", [])
                          if c.get("cell_type") == "code")
            if not has_out:
                never.append(p)
        except Exception:
            pass
    return never


def cell_errors(nb_path: Path) -> list:
    try:
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)
        return [(i, o.get("ename", "?"), o.get("evalue", "")[:120])
                for i, c in enumerate(nb.get("cells", []))
                for o in c.get("outputs", [])
                if o.get("output_type") == "error"]
    except Exception:
        return []


def run_notebook(nb_path: Path) -> dict:
    t0 = time.perf_counter()
    cmd = [
        PY, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        f"--ExecutePreprocessor.kernel_name={KERNEL}",
        f"--ExecutePreprocessor.timeout={TIMEOUT}",
        str(nb_path),
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=TIMEOUT + 90, cwd=str(nb_path.parent)
        )
        elapsed = round(time.perf_counter() - t0, 1)
        if r.returncode == 0:
            errs = cell_errors(nb_path)
            if errs:
                return {"status": "cell_error", "time": elapsed,
                        "cell_errors": errs[:5]}
            return {"status": "success", "time": elapsed}
        err_tail = (r.stderr or "")[-400:] + (r.stdout or "")[-200:]
        return {"status": "error", "time": elapsed, "msg": err_tail}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "time": TIMEOUT}
    except Exception as e:
        return {"status": "exception", "time": 0, "msg": str(e)}


def save():
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Load any saved progress
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            results.update(json.load(f))
        print(f"Loaded {len(results)} previously recorded results.")

    pending = [p for p in find_never_run()
               if str(p.relative_to(REPO)) not in results]
    total = len(pending)
    done_count = len(results)
    print(f"Notebooks to run: {total}  (already done: {done_count})\n")

    counter = [done_count]

    def worker(nb_path: Path):
        res = run_notebook(nb_path)
        rel = str(nb_path.relative_to(REPO))
        with lock:
            results[rel] = res
            counter[0] += 1
            n = counter[0]
            grand = done_count + total
            status_symbol = "OK" if res["status"] == "success" else res["status"].upper()
            print(f"[{n:3d}/{grand}] {rel[:75]:<75} {status_symbol} ({res['time']}s)",
                  flush=True)
            save()
        return rel, res

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(worker, p): p for p in pending}
        for fut in as_completed(futures):
            fut.result()  # re-raise exceptions if any

    # ── summary ──────────────────────────────────────────────────────────────
    all_results = list(results.items())
    ok       = [r for r in all_results if r[1].get("status") == "success"]
    cell_err = [r for r in all_results if r[1].get("status") == "cell_error"]
    errored  = [r for r in all_results if r[1].get("status") == "error"]
    timeout  = [r for r in all_results if r[1].get("status") == "timeout"]

    print(f"\n{'='*70}")
    print(f"TOTAL : {len(all_results)}")
    print(f"  OK          : {len(ok)}")
    print(f"  Cell errors : {len(cell_err)}")
    print(f"  Exec errors : {len(errored)}")
    print(f"  Timeouts    : {len(timeout)}")

    if cell_err:
        print("\n--- CELL ERRORS ---")
        for rel, res in cell_err:
            print(f"  {rel}")
            for ci, ename, ev in res.get("cell_errors", [])[:3]:
                print(f"    Cell {ci}: {ename}: {ev}")

    if errored:
        print("\n--- EXEC ERRORS (last 200 chars) ---")
        for rel, res in errored[:10]:
            print(f"  {rel}")
            print(f"    {res.get('msg','')[-200:]}")

    if timeout:
        print("\n--- TIMEOUTS ---")
        for rel, _ in timeout:
            print(f"  {rel}")

    save()
    print(f"\nResults written to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
