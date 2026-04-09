#!/usr/bin/env python
"""Rebuild Phase 2.1 reports from individual project metrics files."""
import json, sys, os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from scripts.run_all_phase2 import (
    PHASE2_PROJECTS, _generate_summary, _generate_leaderboard,
    _generate_failures, TRAIN_DEFAULTS,
)
from utils.training_common import OUTPUTS_DIR

def _reconstruct_result(slug: str, cfg: dict) -> dict:
    """Reconstruct the result dict from saved metrics."""
    mf = OUTPUTS_DIR / slug / "metrics" / "phase2_metrics.json"
    if not mf.exists():
        return {"status": "MISSING", "slug": slug, "action": cfg["action"],
                "error": "No metrics file found", "time_sec": 0}

    m = json.loads(mf.read_text())
    action = cfg["action"]
    result = {
        "status": m.get("status", "OK"),
        "slug": slug,
        "action": action,
        "model_name": m.get("model_name") or m.get("embedding_model") or cfg.get("model_name", "?"),
    }

    # Time
    result["time_sec"] = m.get("train_runtime_sec", 0) or 0

    # Dataset size
    for key in ("n_train", "n_texts", "n_images"):
        if key in m and m[key]:
            result["dataset_size"] = m[key]
            break
    else:
        result["dataset_size"] = 0

    # Training mode
    if m.get("lora"):
        result["training_mode"] = "LoRA"
    elif action in ("embed_cluster", "caption"):
        result["training_mode"] = "inference"
    elif action == "char_rnn":
        result["training_mode"] = "full"
    else:
        result["training_mode"] = "full"

    # Main metrics (varies by action)
    mm = {}
    if action == "classify":
        tm = m.get("test_metrics", {})
        mm = {
            "accuracy": tm.get("accuracy", 0),
            "f1_weighted": tm.get("f1_weighted", 0),
            "f1_macro": tm.get("f1_macro", 0),
            "f1_micro": tm.get("f1_micro", 0),
        }
    elif action == "classify_multilabel":
        tm = m.get("test_metrics", {})
        mm = {
            "f1_micro": tm.get("f1_micro", 0),
            "f1_macro": tm.get("f1_macro", 0),
            "f1_weighted": tm.get("f1_weighted", 0),
        }
    elif action == "summarize":
        tm = m.get("test_metrics", m.get("val_metrics", {}))
        mm = {
            "rouge1": tm.get("rouge1", 0),
            "rouge2": tm.get("rouge2", 0),
            "rougeL": tm.get("rougeL", 0),
        }
    elif action == "translate":
        tm = m.get("test_metrics", {})
        mm = {
            "bleu": tm.get("bleu", 0),
            "chrf": tm.get("chrf", 0),
        }
    elif action == "embed_cluster":
        cl = m.get("clustering", {})
        mm = {
            "method": cl.get("method", "?"),
            "n_clusters": cl.get("n_clusters", 0),
            "n_noise": cl.get("n_noise", 0),
            "silhouette": cl.get("silhouette", 0),
            "calinski_harabasz": cl.get("calinski_harabasz", 0),
        }
    elif action == "caption":
        tm = m.get("test_metrics", m.get("metrics", {}))
        mm = {
            "rougeL": tm.get("rougeL", tm.get("rouge_l", 0)),
            "bleu": tm.get("bleu", tm.get("bleu_4", 0)),
        }
    elif action == "char_rnn":
        mm = {
            "best_val_loss": m.get("best_val_loss", 0),
        }

    result["main_metrics"] = mm
    result["val_metrics"] = m.get("val_metrics", {})

    return result


all_results = {}
for slug, cfg in PHASE2_PROJECTS.items():
    all_results[slug] = _reconstruct_result(slug, cfg)

# Print summary table
print("\n" + "=" * 95)
print(f"{'#':>3} | {'Slug':<42} | {'Status':<7} | {'Key Metric':<25} | {'Time':>7}")
print("-" * 95)
from scripts.run_all_phase2 import _metric_str
for i, (slug, r) in enumerate(all_results.items(), 1):
    st = r.get("status", "?")
    act = r.get("action", "?")
    mm = r.get("main_metrics", {})
    ms = _metric_str(act, mm) if st == "OK" else f"[{st}]"
    t = r.get("time_sec", 0)
    print(f"{i:>3} | {slug:<42} | {st:<7} | {ms:<25} | {t:>6.0f}s")
print("=" * 95)

ok = sum(1 for r in all_results.values() if r.get("status") == "OK")
fail = len(all_results) - ok
total_t = sum(r.get("time_sec", 0) for r in all_results.values())
print(f"Total: {ok} OK, {fail} FAILED, {total_t:.0f}s ({total_t/60:.1f} min)")

# Generate reports
_generate_summary(all_results)
_generate_leaderboard(all_results)
_generate_failures(all_results)
print("\nReports regenerated in reports/")
