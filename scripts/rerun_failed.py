#!/usr/bin/env python
"""Re-run the 2 failed Phase 2.1 projects after fixes."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scripts.run_all_phase2 import (
    _process_project, PHASE2_PROJECTS, _generate_summary,
    _generate_leaderboard, _generate_failures, REPORTS,
)
from utils.training_common import cleanup_gpu, seed_everything, OUTPUTS_DIR
from pathlib import Path

seed_everything(42)

FAILED_SLUGS = [
    "kaggle-survey-questions-clustering",
    "bbc-articles-summarization",
]

# Load existing results from individual project metrics
WORKSPACE = Path(__file__).resolve().parent.parent
all_results = {}

for slug, cfg in PHASE2_PROJECTS.items():
    mf = OUTPUTS_DIR / slug / "metrics" / "phase2_metrics.json"
    if mf.exists():
        all_results[slug] = json.loads(mf.read_text())
        all_results[slug]["slug"] = slug
        all_results[slug]["action"] = cfg["action"]
    else:
        all_results[slug] = {"status": "MISSING", "slug": slug, "action": cfg["action"]}

# Re-run failed projects
for slug in FAILED_SLUGS:
    print("=" * 70)
    print(f"Re-running {slug}...")
    print("=" * 70)
    cfg = PHASE2_PROJECTS[slug]
    try:
        result = _process_project(slug, cfg)
        all_results[slug] = result
        print(f"  => {result.get('status')} | {json.dumps(result.get('main_metrics', {}), default=str)[:200]}")
    except Exception as e:
        import traceback
        print(f"  => FAILED: {e}")
        all_results[slug] = {"status": "FAILED", "slug": slug, "action": cfg["action"],
                             "error": str(e), "traceback": traceback.format_exc()}
    cleanup_gpu()

# Regenerate reports with updated results
print("\n" + "=" * 70)
print("Regenerating reports...")
_generate_summary(all_results)
_generate_leaderboard(all_results)
_generate_failures(all_results)

ok = sum(1 for r in all_results.values() if r.get("status") == "OK")
fail = len(all_results) - ok
print(f"\nFinal: {ok} OK, {fail} FAILED out of {len(all_results)} projects")
print("Reports updated in reports/")
