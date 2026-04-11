#!/usr/bin/env python3
"""
master_audit.py  –  Full-repository MLOps audit orchestrator
=============================================================
Runs the complete production-grade audit:
  1. Inject MLOps bootstrap cells into every notebook
  2. Execute every notebook category by category
  3. Generate summary.json, leaderboard.json, final status report
  4. Exit 1 if any notebook scores < 70 (hard failure)

Usage:
  python master_audit.py                         # all notebooks
  python master_audit.py --category "NLP"        # one category
  python master_audit.py --skip-inject           # skip MLOps injection pass
  python master_audit.py --resume                # skip already-passing notebooks
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.resolve()
REPORTS_DIR = REPO_ROOT / "reports"
LOGS_DIR    = REPO_ROOT / "logs"
PYTHON      = str(REPO_ROOT / "venv" / "Scripts" / "python.exe")

CATEGORIES_IN_ORDER = [
    # Fast  (<5min each typically)
    "Data Analysis",
    "Clustering",
    "Conceptual",
    # Medium (5-20min each)
    "Classification",
    "Regression",
    "Anomaly detection",
    "Recommendation Systems",
    "Time Series Analysis",
    # Slower (model downloads, heavy training)
    "NLP",
    "Computer Vision",
    "Deep Learning",
    "Reinforcement Learning",
    "Speech and Audio",
    # LLM-based (need Ollama)
    "RAG",
    "Advance RAG",
    "GenAI",
    "LangGraph",
    "LlamaIndex",
    "Agents with tools",
    "AgenticAI",
    "Advanced agentic",
    "CrewAI",
    "Fine tuning",
    "100_Local_AI_Projects",
]


def run_step(cmd: list[str], log_file: Path, label: str) -> int:
    print(f"\n{'='*60}", flush=True)
    print(f"  STARTING: {label}", flush=True)
    print(f"  Log: {log_file}", flush=True)
    print(f"{'='*60}", flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    with log_file.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(REPO_ROOT))
    # Print tail of log after completion
    lines = log_file.read_text(encoding="utf-8").splitlines()
    for line in lines[-20:]:
        print(f"  {line}", flush=True)
    return proc.returncode


def read_summary() -> dict:
    try:
        return json.loads((REPORTS_DIR / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--category",    default=None, help="Only audit this category")
    parser.add_argument("--skip-inject", action="store_true")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip notebooks that already have a passing report")
    parser.add_argument("--workers",     type=int, default=1)
    args = parser.parse_args()

    categories = [args.category] if args.category else CATEGORIES_IN_ORDER
    t_start = time.time()

    # ── Phase 1: inject MLOps bootstrap cells ─────────────────────────────────
    if not args.skip_inject:
        print("\n[Phase 1] Injecting MLOps bootstrap cells …", flush=True)
        inject_cmd = [PYTHON, str(REPO_ROOT / "inject_mlops_cell.py")]
        if args.category:
            inject_cmd += ["--category", args.category]
        run_step(inject_cmd, LOGS_DIR / "inject_mlops.txt", "MLOps injection")

    # ── Phase 2: audit each category ─────────────────────────────────────────
    print("\n[Phase 2] Auditing notebooks category by category …", flush=True)
    all_ok = True
    for cat in categories:
        cat_safe = cat.replace(" ", "_").replace("/", "_")
        log_file = LOGS_DIR / f"audit_{cat_safe}.txt"
        cmd = [PYTHON, "-u", str(REPO_ROOT / "run_ml_audit.py"),
               "--category", cat,
               "--workers", str(args.workers)]
        if args.resume:
            cmd.append("--resume")
        rc = run_step(cmd, log_file, f"Audit: {cat}")
        if rc != 0:
            print(f"  [WARN] category '{cat}' exited with code {rc}", flush=True)
            all_ok = False
        time.sleep(1)   # brief pause between categories for OS resource cleanup

    # ── Phase 3: print final summary ─────────────────────────────────────────
    summary = read_summary()
    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print("  MASTER AUDIT COMPLETE", flush=True)
    print(f"  elapsed:         {elapsed/60:.1f} min", flush=True)
    print(f"  total notebooks: {summary.get('total_notebooks',0)}", flush=True)
    print(f"  passed:          {summary.get('passed',0)}", flush=True)
    print(f"  failed:          {summary.get('failed',0)}", flush=True)
    print(f"  avg score:       {summary.get('avg_score',0):.1f}", flush=True)
    print(f"  PRODUCTION_READY:{summary.get('production_ready',0)}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\n  Reports: {REPORTS_DIR}", flush=True)
    print(f"  Leaderboard: {REPORTS_DIR/'leaderboard.json'}", flush=True)

    # Exit 1 if hard failures exist (score < 70)
    hard_fails = [r for r in summary.get("ranking_asc", []) if r.get("score", 100) < 70]
    if hard_fails:
        print(f"\n  [FAIL] {len(hard_fails)} notebooks scored below 70:", flush=True)
        for hf in hard_fails[:20]:
            print(f"    - {hf['notebook']} (score={hf['score']})", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
