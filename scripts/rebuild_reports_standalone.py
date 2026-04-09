#!/usr/bin/env python
"""Rebuild Phase 2.1 reports from individual metrics files (standalone)."""
import csv, json, sys, os
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
OUTPUTS = WORKSPACE / "outputs"
REPORTS = WORKSPACE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

PROJECTS = [
    ("e-commerce-clothing-reviews", "classify"),
    ("trip-advisor-hotel-reviews", "classify"),
    ("cyberbullying-classification", "classify"),
    ("e-commerce-product-classification", "classify"),
    ("economic-news-articles", "classify"),
    ("fake-news-detection", "classify"),
    ("news-headline-classification", "classify"),
    ("paper-subject-prediction", "classify"),
    ("review-classification", "classify"),
    ("spam-message-detection", "classify"),
    ("twitter-sentiment-analysis", "classify"),
    ("toxic-comment-classification", "classify_multilabel"),
    ("world-war-i-letters", "embed_cluster"),
    ("kaggle-survey-questions-clustering", "embed_cluster"),
    ("medium-articles-clustering", "embed_cluster"),
    ("newsgroups-posts-clustering", "embed_cluster"),
    ("stories-clustering", "embed_cluster"),
    ("bbc-articles-summarization", "summarize"),
    ("english-to-french-translation", "translate"),
    ("automated-image-captioning", "caption"),
    ("name-generate-from-languages", "char_rnn"),
]

def load_result(slug, action):
    mf = OUTPUTS / slug / "metrics" / "phase2_metrics.json"
    if not mf.exists():
        return {"status": "MISSING", "slug": slug, "action": action,
                "model_name": "?", "training_mode": "?", "dataset_size": 0,
                "main_metrics": {}, "time_sec": 0, "error": "No metrics file"}
    m = json.loads(mf.read_text(encoding="utf-8"))
    r = {"status": m.get("status", "OK"), "slug": slug, "action": action}
    r["model_name"] = m.get("model_name") or m.get("embedding_model") or m.get("model") or "?"
    r["time_sec"] = m.get("train_runtime_sec") or 0

    # dataset size
    for k in ("n_train", "n_texts", "n_images_total", "n_images", "n_names"):
        if k in m and m[k]:
            r["dataset_size"] = m[k]; break
    else:
        r["dataset_size"] = 0

    # training mode
    if m.get("lora"):
        r["training_mode"] = "LoRA"
    elif action in ("embed_cluster", "caption"):
        r["training_mode"] = "inference"
    else:
        r["training_mode"] = "full"

    # main metrics
    mm = {}
    if action == "classify":
        tm = m.get("test_metrics", {})
        mm = {k: tm.get(k, 0) for k in ("accuracy", "f1_weighted", "f1_macro", "f1_micro")}
    elif action == "classify_multilabel":
        tm = m.get("test_metrics", {})
        mm = {k: tm.get(k, 0) for k in ("f1_micro", "f1_macro", "f1_weighted")}
    elif action == "summarize":
        tm = m.get("test_metrics", m.get("val_metrics", {}))
        mm = {k: tm.get(k, 0) for k in ("rouge1", "rouge2", "rougeL")}
    elif action == "translate":
        tm = m.get("test_metrics", {})
        mm = {"bleu": tm.get("bleu", 0), "chrf": tm.get("chrf", 0)}
    elif action == "embed_cluster":
        cl = m.get("clustering", {})
        mm = {k: cl.get(k, 0) for k in ("method", "n_clusters", "n_noise", "silhouette", "calinski_harabasz")}
    elif action == "caption":
        tm = m.get("eval_metrics", m.get("test_metrics", m.get("metrics", {})))
        mm = {"rougeL": tm.get("rougeL", tm.get("rouge_l", 0)),
              "bleu": tm.get("bleu", tm.get("bleu1_approx", tm.get("bleu_4", 0)))}
    elif action == "char_rnn":
        mm = {"best_val_loss": m.get("best_val_loss", 0)}
    r["main_metrics"] = mm
    r["val_metrics"] = m.get("val_metrics", {})
    return r


def metric_str(action, mm):
    if action == "classify":
        return f"F1w={mm.get('f1_weighted', 0):.4f}"
    if action == "classify_multilabel":
        return f"F1mi={mm.get('f1_micro', 0):.4f}"
    if action == "summarize":
        return f"R-L={mm.get('rougeL', 0):.4f}"
    if action == "translate":
        return f"BLEU={mm.get('bleu', 0):.1f}"
    if action == "embed_cluster":
        return f"Sil={mm.get('silhouette', 0):.4f}"
    if action == "caption":
        return f"R-L={mm.get('rougeL', 0):.4f}"
    if action == "char_rnn":
        return f"VL={mm.get('best_val_loss', 0):.4f}"
    return "?"


# Build all results
all_results = {}
for slug, action in PROJECTS:
    all_results[slug] = load_result(slug, action)

# Print table
print("\n" + "=" * 100)
print(f"{'#':>3} | {'Project':<42} | {'Status':<7} | {'Mode':<9} | {'Size':>6} | {'Main Metric':<20} | {'Time':>7}")
print("-" * 100)
for i, (slug, r) in enumerate(all_results.items(), 1):
    st = r["status"]
    ms = metric_str(r["action"], r["main_metrics"]) if st == "OK" else f"[{st}]"
    print(f"{i:>3} | {slug:<42} | {st:<7} | {r.get('training_mode','?'):<9} | {r.get('dataset_size',0):>6} | {ms:<20} | {r.get('time_sec',0):>6.0f}s")
print("=" * 100)
ok = sum(1 for r in all_results.values() if r["status"] == "OK")
fail = len(all_results) - ok
total_t = sum(r.get("time_sec", 0) for r in all_results.values())
print(f"Total: {ok} OK, {fail} FAILED, {total_t:.0f}s ({total_t/60:.1f} min)")

# Generate summary MD
ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
L = [
    "# Phase 2.1 -- Training Results Summary\n\n",
    f"Generated: {ts}\n\n",
    "## Overview\n\n",
    "| Metric | Value |\n|--------|-------|\n",
    f"| Total projects | {len(all_results)} |\n",
    f"| Succeeded | {ok} |\n",
    f"| Failed | {fail} |\n",
    f"| Total time | {total_t:.0f}s ({total_t/60:.1f} min) |\n\n",
    "## Results Table\n\n",
    "| # | Project | Action | Model | Mode | Size | Main Metric | Time |\n",
    "|---|---------|--------|-------|------|------|-------------|------|\n",
]
for i, (slug, r) in enumerate(all_results.items(), 1):
    st = r["status"]
    model = (r.get("model_name") or "?").split("/")[-1][:28]
    mode = r.get("training_mode", "?")
    size = r.get("dataset_size", 0)
    t = r.get("time_sec", 0)
    ms = metric_str(r["action"], r["main_metrics"]) if st == "OK" else f"**{st}**"
    L.append(f"| {i} | {slug} | {r['action']} | {model} | {mode} | {size} | {ms} | {t:.0f}s |\n")
L.append(f"\n**{ok} succeeded, {fail} failed, total {total_t:.0f}s**\n\n")

# Per-project detail
L.append("## Per-Project Details\n\n")
for slug, r in all_results.items():
    L.append(f"### {slug}\n\n")
    L.append(f"- **Status:** {r['status']}\n")
    L.append(f"- **Action:** {r['action']}\n")
    L.append(f"- **Model:** `{r.get('model_name', '?')}`\n")
    L.append(f"- **Training mode:** {r.get('training_mode', '?')}\n")
    L.append(f"- **Dataset size:** {r.get('dataset_size', 0)}\n")
    L.append(f"- **Time:** {r.get('time_sec', 0):.1f}s\n")
    mm = r.get("main_metrics", {})
    if mm:
        L.append(f"- **Metrics:** `{json.dumps(mm, default=str)[:200]}`\n")
    vm = r.get("val_metrics", {})
    if vm:
        L.append(f"- **Val metrics:** `{json.dumps(vm, default=str)[:200]}`\n")
    if r.get("error"):
        L.append(f"- **Error:** `{r['error']}`\n")
    L.append(f"- **Outputs:** `outputs/{slug}/`\n\n")

(REPORTS / "phase2_summary.md").write_text("".join(L), encoding="utf-8")
print("Wrote reports/phase2_summary.md")

# Generate leaderboard CSV
rows = []
for slug, r in all_results.items():
    if r["status"] != "OK":
        continue
    act = r["action"]
    mm = r.get("main_metrics", {})
    row = {"slug": slug, "action": act, "model": r.get("model_name", "?"),
           "mode": r.get("training_mode", "?"), "dataset_size": r.get("dataset_size", 0)}
    if act == "classify":
        row.update({"metric_name": "f1_macro", "metric_value": round(mm.get("f1_macro", 0), 4),
                    "accuracy": round(mm.get("accuracy", 0), 4), "f1_weighted": round(mm.get("f1_weighted", 0), 4)})
    elif act == "classify_multilabel":
        row.update({"metric_name": "f1_micro", "metric_value": round(mm.get("f1_micro", 0), 4),
                    "f1_macro": round(mm.get("f1_macro", 0), 4)})
    elif act == "summarize":
        row.update({"metric_name": "rougeL", "metric_value": round(mm.get("rougeL", 0), 4),
                    "rouge1": round(mm.get("rouge1", 0), 4)})
    elif act == "translate":
        row.update({"metric_name": "bleu", "metric_value": round(mm.get("bleu", 0), 2),
                    "chrf": round(mm.get("chrf", 0), 2)})
    elif act == "embed_cluster":
        row.update({"metric_name": "silhouette", "metric_value": round(mm.get("silhouette", 0), 4)})
    elif act == "caption":
        row.update({"metric_name": "rougeL", "metric_value": round(mm.get("rougeL", 0), 4)})
    elif act == "char_rnn":
        row.update({"metric_name": "val_loss", "metric_value": round(mm.get("best_val_loss", 0), 4)})
    rows.append(row)

rows.sort(key=lambda x: x.get("metric_value", 0), reverse=True)
path = REPORTS / "phase2_leaderboard.csv"
if rows:
    all_keys = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
print("Wrote reports/phase2_leaderboard.csv")

# Generate failures
FL = [f"# Phase 2.1 Failures\n\nGenerated: {ts}\n\n"]
failures = {s: r for s, r in all_results.items() if r["status"] != "OK"}
if failures:
    FL.append(f"**{len(failures)} project(s) failed:**\n\n")
    for slug, r in failures.items():
        FL.append(f"## {slug}\n\n")
        FL.append(f"- **Action:** {r.get('action', '?')}\n")
        FL.append(f"- **Error:** {r.get('error', 'Unknown')}\n\n")
else:
    FL.append("No failures -- all 21 projects completed successfully!\n")
(REPORTS / "phase2_failures.md").write_text("".join(FL), encoding="utf-8")
print("Wrote reports/phase2_failures.md")
print("\nDone!")
