"""Phase 3 – Quality + Stability runner.

Usage:
    python -m scripts.run_phase3 --project <slug> [--force]
    python -m scripts.run_phase3 --bad-only [--force]
    python -m scripts.run_phase3 --all [--force]

Loads config/base_config.yaml + config/projects/<slug>.yaml overrides,
then delegates to the same training modules used in Phase 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

from utils.training_common import (
    SEED,
    cleanup_gpu,
    ensure_output_dirs,
    get_device,
    save_json,
    seed_everything,
)
from utils.logger import get_logger

logger = get_logger(__name__)

DATA_ROOT   = WORKSPACE / "data"
OUTPUTS_DIR = WORKSPACE / "outputs"
REPORTS     = WORKSPACE / "reports"
CONFIG_DIR  = WORKSPACE / "config"

# Import the project registry built in Phase 1-2
try:
    from utils.dataset_finder import PROJECT_REGISTRY
except ImportError:
    PROJECT_REGISTRY = {}

# ---------------------------------------------------------------------------
# BAD slugs from Phase 2.1  (F1w < 0.40  or  F1mi < 0.20)
# ---------------------------------------------------------------------------
BAD_SLUGS = [
    "trip-advisor-hotel-reviews",
    "cyberbullying-classification",
    "e-commerce-product-classification",
    "economic-news-articles",
    "fake-news-detection",
    "news-headline-classification",
    "paper-subject-prediction",
    "review-classification",
    "twitter-sentiment-analysis",
    "toxic-comment-classification",
]

# ---------------------------------------------------------------------------
# Project definitions (same as run_all_phase2.py)
# ---------------------------------------------------------------------------
from scripts.run_all_phase2 import (
    PHASE2_PROJECTS,
    TRAIN_DEFAULTS,
    _find_raw_dir,
    _load_data,
    _find_text_col,
    _find_target_col,
    _save_extra_outputs,
)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_project_config(slug: str) -> dict:
    """Merge base_config.yaml + projects/<slug>.yaml overrides."""
    base_path = CONFIG_DIR / "base_config.yaml"
    base = {}
    if base_path.exists():
        base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

    proj_path = CONFIG_DIR / "projects" / f"{slug}.yaml"
    overrides: dict = {}
    if proj_path.exists():
        overrides = yaml.safe_load(proj_path.read_text(encoding="utf-8")) or {}

    # Flatten nested sections for easy consumption
    training = base.get("training", {})
    merged = {**TRAIN_DEFAULTS, **training}

    # Apply task-specific defaults based on project action
    cfg = PHASE2_PROJECTS.get(slug, {})
    action = cfg.get("action", "classify")
    if action == "classify":
        merged.update(base.get("classification", {}))
    elif action == "classify_multilabel":
        merged.update(base.get("multilabel", {}))
    elif action == "embed_cluster":
        merged.update(base.get("clustering", {}))
    elif action == "summarize":
        merged.update(base.get("summarization", {}))
    elif action == "translate":
        merged.update(base.get("translation", {}))
    elif action == "caption":
        merged.update(base.get("captioning", {}))
    elif action == "char_rnn":
        merged.update(base.get("generation", {}))

    # Apply per-project overrides last (highest priority)
    merged.update(overrides)
    return merged


# ---------------------------------------------------------------------------
# Phase 2 metrics loader (for before/after comparison)
# ---------------------------------------------------------------------------

def _load_phase2_metric(slug: str) -> dict | None:
    """Load the Phase 2.1 metrics file for a project."""
    p = OUTPUTS_DIR / slug / "metrics" / "phase2_metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------------
# Single-project runner
# ---------------------------------------------------------------------------

def run_single_project(slug: str, *, force: bool = False) -> dict:
    """Run Phase 3 training for one project."""
    t0 = time.time()
    cfg = PHASE2_PROJECTS.get(slug)
    if cfg is None:
        return {"status": "FAILED", "slug": slug, "error": f"Unknown slug: {slug}", "time_sec": 0}

    action = cfg["action"]
    raw_dir = _find_raw_dir(slug)
    if raw_dir is None:
        return {"status": "FAILED", "slug": slug, "action": action,
                "error": "Data directory not found", "time_sec": 0}

    dirs = ensure_output_dirs(slug)
    outputs_dir = OUTPUTS_DIR / slug

    # Build merged config
    proj_config = _load_project_config(slug)
    config = {**proj_config}
    # Also keep original cfg keys (text_col, target_col, data_loader, etc.)
    config.update({k: v for k, v in cfg.items()
                   if k not in ("action", "data_file", "data_loader",
                                "csv_kwargs", "preprocess_target")})

    try:
        if action == "classify":
            data = _load_data(slug, cfg, raw_dir)
            df = data["df"]
            text_col = _find_text_col(df, cfg.get("text_col", "text"))
            target_col = _find_target_col(df, cfg.get("target_col", "label"))
            if target_col is None:
                return {"status": "FAILED", "slug": slug, "action": action, "time_sec": 0,
                        "error": f"No target column in {list(df.columns)}"}
            if cfg.get("preprocess_target") == "first_term":
                df[target_col] = (df[target_col].astype(str)
                                  .str.strip("[]' ").str.split(",").str[0].str.strip("' "))
            config.update({"df": df, "text_col": text_col, "target_col": target_col,
                           "run_kfold": True, "kfold_n": 5})
            from utils.train_text_classifier import run_project
            result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

        elif action == "classify_multilabel":
            data = _load_data(slug, cfg, raw_dir)
            df = data["df"]
            text_col = _find_text_col(df, cfg.get("text_col", "comment_text"))
            target_cols = cfg.get("target_cols", [])
            missing = [c for c in target_cols if c not in df.columns]
            if missing:
                return {"status": "FAILED", "slug": slug, "action": action, "time_sec": 0,
                        "error": f"Missing cols {missing}"}
            config.update({"df": df, "text_col": text_col,
                           "target_cols": target_cols, "multilabel": True})
            from utils.train_text_classifier import run_project
            result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

        elif action == "embed_cluster":
            data = _load_data(slug, cfg, raw_dir)
            if "texts" in data:
                texts = data["texts"]
            else:
                df = data["df"]
                tc = _find_text_col(df, cfg.get("text_col", "_auto"))
                texts = df[tc].dropna().astype(str).tolist()
            if len(texts) < 5 and "df" in data:
                df = data["df"]
                all_texts = []
                for _, row in df.iterrows():
                    for val in row:
                        s = str(val).strip()
                        if len(s) > 10 and not s.replace(".", "").replace("-", "").isdigit():
                            all_texts.append(s)
                if len(all_texts) > len(texts):
                    texts = all_texts
            config["texts"] = texts
            from utils.embeddings_and_topics import run_project
            result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

        elif action == "summarize":
            config["raw_dir"] = raw_dir
            from utils.train_summarizer import run_project
            result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

        elif action == "translate":
            config["raw_dir"] = raw_dir
            from utils.train_translator import run_project
            result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

        elif action == "caption":
            images_dir = None
            for cand in [raw_dir / "images", raw_dir / "Images", raw_dir]:
                if cand.is_dir() and (list(cand.glob("*.jpg")) or list(cand.glob("*.png"))):
                    images_dir = cand
                    break
            captions_path = None
            for nm in ["captions.txt", "captions.csv", "token.txt", "results.csv"]:
                c = raw_dir / nm
                if c.exists():
                    captions_path = c
                    break
            config.update({"images_dir": images_dir, "captions_path": captions_path})
            from utils.captioning import run_project
            result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

        elif action == "char_rnn":
            names_dir = None
            for cand in [raw_dir / "names", raw_dir]:
                if cand.is_dir() and any(cand.glob("*.txt")):
                    names_dir = cand
                    break
            if names_dir is None:
                info = PROJECT_REGISTRY.get(slug, {})
                pdir = WORKSPACE / info.get("dir", "")
                for cand in [pdir / "names", pdir]:
                    if cand.is_dir() and any(cand.glob("*.txt")):
                        names_dir = cand
                        break
            if names_dir is None:
                return {"status": "FAILED", "slug": slug, "action": action,
                        "error": "names dir not found", "time_sec": 0}
            config.update({"names_dir": names_dir})
            from utils.training_common import run_char_rnn_project as rp
            result = rp(slug, raw_dir, None, None, outputs_dir, config, force=force)

        else:
            return {"status": "FAILED", "slug": slug, "action": action,
                    "error": f"Unknown action: {action}", "time_sec": 0}

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[%s] FAILED: %s", slug, exc)
        return {"status": "FAILED", "slug": slug, "action": action,
                "error": str(exc), "traceback": tb, "time_sec": round(time.time() - t0, 1)}

    result["slug"] = slug
    result["action"] = action
    result["time_sec"] = round(time.time() - t0, 1)
    _save_extra_outputs(slug, result, dirs)
    return result


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------

def _primary_metric(action: str, mm: dict) -> tuple[str, float]:
    """Return (metric_name, metric_value) for the primary metric."""
    if action == "classify":
        return "f1_weighted", float(mm.get("f1_weighted", mm.get("f1", 0)))
    if action == "classify_multilabel":
        return "f1_micro", float(mm.get("f1_micro", 0))
    if action == "summarize":
        return "rougeL", float(mm.get("rougeL", 0))
    if action == "translate":
        return "bleu", float(mm.get("bleu", 0))
    if action == "embed_cluster":
        return "silhouette", float(mm.get("silhouette", 0))
    if action == "caption":
        return "rougeL", float(mm.get("rougeL", 0))
    if action == "char_rnn":
        return "val_loss", float(mm.get("best_val_loss", 0))
    return "unknown", 0.0


def _metric_str(action: str, mm: dict) -> str:
    name, val = _primary_metric(action, mm)
    if name == "bleu":
        return f"BLEU={val:.1f}"
    return f"{name}={val:.4f}"


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

def _generate_phase3_reports(all_results: dict, phase2_metrics: dict) -> None:
    """Generate phase3_summary.md and phase3_leaderboard.csv."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    ok = sum(1 for r in all_results.values() if r.get("status") == "OK")
    fail = len(all_results) - ok
    total_t = sum(r.get("time_sec", 0) for r in all_results.values())

    # ---- Summary MD -------------------------------------------------------
    L = [
        "# Phase 3.0 — Quality & Stability Upgrades\n\n",
        f"Generated: {ts}\n\n",
        "## Changes Applied\n\n",
        "1. **Class weight clamping** — weights clamped to [0.5, 5.0], normalised mean=1\n",
        "2. **Gradient clipping** — `max_grad_norm=1.0`\n",
        "3. **Label smoothing** — `label_smoothing=0.05`\n",
        "4. **Cosine LR schedule** — `lr_scheduler_type=cosine`\n",
        "5. **Focal loss** — `(1-p_t)^gamma` for many-class problems (gamma=1.5-2.0)\n",
        "6. **Multilabel fix** — BCEWithLogitsLoss + clamped pos_weight + per-label threshold tuning\n",
        "7. **Per-project YAML configs** — `config/projects/<slug>.yaml`\n\n",
        "## Overview\n\n",
        f"| Metric | Value |\n|--------|-------|\n",
        f"| Re-trained projects | {len(all_results)} |\n",
        f"| Succeeded | {ok} |\n",
        f"| Failed | {fail} |\n",
        f"| Total time | {total_t:.0f}s ({total_t/60:.1f} min) |\n\n",
        "## Before / After Comparison\n\n",
        "| # | Project | Action | Phase 2 | Phase 3 | Delta | Status |\n",
        "|---|---------|--------|---------|---------|-------|--------|\n",
    ]

    improvements = 0
    for i, (slug, r) in enumerate(all_results.items(), 1):
        act = r.get("action", "?")
        st = r.get("status", "?")
        mm = r.get("main_metrics", {})

        # Phase 2 baseline
        p2 = phase2_metrics.get(slug, {})
        p2_mm = p2.get("test_metrics", p2.get("clustering", p2.get("eval_metrics", {})))
        if not p2_mm and "main_metrics" in p2:
            p2_mm = p2["main_metrics"]

        p2_name, p2_val = _primary_metric(act, p2_mm)
        p3_name, p3_val = _primary_metric(act, mm) if st == "OK" else (p2_name, 0.0)

        delta = p3_val - p2_val
        delta_str = f"{delta:+.4f}" if abs(delta) < 100 else f"{delta:+.1f}"
        improved = "YES" if delta > 0.01 else ("same" if abs(delta) < 0.01 else "regressed")
        if delta > 0.01:
            improvements += 1

        p2_str = f"{p2_val:.4f}" if abs(p2_val) < 100 else f"{p2_val:.1f}"
        p3_str = _metric_str(act, mm) if st == "OK" else f"**{st}**"

        L.append(f"| {i} | {slug} | {act} | {p2_str} | {p3_str} | {delta_str} | {improved} |\n")

    L.append(f"\n**{improvements}/{len(all_results)} improved**\n\n")

    # Per-project details
    L.append("## Per-Project Details\n\n")
    for slug, r in all_results.items():
        L.append(f"### {slug}\n\n")
        L.append(f"- **Status:** {r.get('status', '?')}\n")
        L.append(f"- **Action:** {r.get('action', '?')}\n")
        L.append(f"- **Model:** `{r.get('model_name', '?')}`\n")
        L.append(f"- **Time:** {r.get('time_sec', 0):.1f}s\n")
        mm = r.get("main_metrics", {})
        if mm:
            L.append(f"- **Test metrics:** `{json.dumps(mm, default=str)[:250]}`\n")
        vm = r.get("val_metrics", {})
        if vm:
            L.append(f"- **Val metrics:** `{json.dumps(vm, default=str)[:200]}`\n")
        if r.get("error"):
            L.append(f"- **Error:** `{r['error']}`\n")
        L.append("\n")

    (REPORTS / "phase3_summary.md").write_text("".join(L), encoding="utf-8")
    logger.info("Wrote reports/phase3_summary.md")

    # ---- Leaderboard CSV --------------------------------------------------
    rows: list[dict] = []
    for slug, r in all_results.items():
        act = r.get("action", "?")
        mm = r.get("main_metrics", {})
        p3_name, p3_val = _primary_metric(act, mm) if r.get("status") == "OK" else ("?", 0)

        p2 = phase2_metrics.get(slug, {})
        p2_mm = p2.get("test_metrics", p2.get("clustering", p2.get("eval_metrics", {})))
        if not p2_mm and "main_metrics" in p2:
            p2_mm = p2["main_metrics"]
        p2_name, p2_val = _primary_metric(act, p2_mm)

        rows.append({
            "slug": slug,
            "action": act,
            "model": r.get("model_name", "?"),
            "metric_name": p3_name,
            "phase2_value": round(p2_val, 4),
            "phase3_value": round(p3_val, 4),
            "delta": round(p3_val - p2_val, 4),
            "status": r.get("status", "?"),
            "time_sec": r.get("time_sec", 0),
        })

    rows.sort(key=lambda r: r.get("delta", 0), reverse=True)
    path = REPORTS / "phase3_leaderboard.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    logger.info("Wrote reports/phase3_leaderboard.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--project", type=str, help="Single project slug to run")
    group.add_argument("--bad-only", action="store_true", help="Re-run only bad projects from Phase 2.1")
    group.add_argument("--all", action="store_true", help="Re-run all 21 projects")
    parser.add_argument("--force", action="store_true", help="Force re-run even if metrics exist")
    args = parser.parse_args()

    seed_everything(SEED)
    device = get_device()

    if args.project:
        slugs = [args.project]
    elif args.bad_only:
        slugs = BAD_SLUGS
    else:
        slugs = list(PHASE2_PROJECTS.keys())

    logger.info("=" * 70)
    logger.info("Phase 3.0 — Quality & Stability Upgrades")
    logger.info("  Device: %s", device)
    logger.info("  Projects: %d  (%s)", len(slugs), "bad-only" if args.bad_only else ("single" if args.project else "all"))
    logger.info("  Force: %s", args.force)
    logger.info("=" * 70)

    # Load Phase 2 metrics for comparison
    phase2_metrics: dict[str, dict] = {}
    for slug in slugs:
        m = _load_phase2_metric(slug)
        if m:
            phase2_metrics[slug] = m

    all_results: dict[str, dict] = {}

    for i, slug in enumerate(slugs, 1):
        logger.info("-" * 70)
        logger.info("[%d/%d] %s", i, len(slugs), slug)
        logger.info("-" * 70)

        try:
            result = run_single_project(slug, force=args.force)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("[%s] UNHANDLED ERROR: %s", slug, exc)
            result = {"status": "FAILED", "slug": slug, "action": "?",
                      "error": str(exc), "traceback": tb, "time_sec": 0}

        all_results[slug] = result

        st = result.get("status", "UNKNOWN")
        mm = result.get("main_metrics", {})
        act = result.get("action", "?")
        ms = _metric_str(act, mm) if st == "OK" else f"[{st}]"
        logger.info("  => %s  (%.1fs)  %s", st, result.get("time_sec", 0), ms)

        # Robust GPU cleanup — even if CUDA is in error state
        try:
            cleanup_gpu()
        except Exception:
            logger.warning("cleanup_gpu failed; attempting device reset")
            try:
                import torch
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ---- Reports ----------------------------------------------------------
    _generate_phase3_reports(all_results, phase2_metrics)

    # ---- Terminal summary -------------------------------------------------
    ok = sum(1 for r in all_results.values() if r.get("status") == "OK")
    fail = len(all_results) - ok
    total_t = sum(r.get("time_sec", 0) for r in all_results.values())

    print("\n" + "=" * 100)
    print(f"{'#':>3} | {'Slug':<42} | {'Status':<7} | {'Phase2':>12} | {'Phase3':>12} | {'Delta':>8} | {'Time':>7}")
    print("-" * 100)
    for i, (slug, r) in enumerate(all_results.items(), 1):
        st = r.get("status", "?")
        act = r.get("action", "?")
        mm = r.get("main_metrics", {})

        p2 = phase2_metrics.get(slug, {})
        p2_mm = p2.get("test_metrics", p2.get("clustering", p2.get("eval_metrics", {})))
        if not p2_mm and "main_metrics" in p2:
            p2_mm = p2["main_metrics"]
        _, p2_val = _primary_metric(act, p2_mm)
        _, p3_val = _primary_metric(act, mm) if st == "OK" else ("?", 0)

        p2_s = f"{p2_val:.4f}"
        p3_s = f"{p3_val:.4f}" if st == "OK" else f"[{st}]"
        delta = p3_val - p2_val
        delta_s = f"{delta:+.4f}" if st == "OK" else "N/A"
        t = r.get("time_sec", 0)

        print(f"{i:>3} | {slug:<42} | {st:<7} | {p2_s:>12} | {p3_s:>12} | {delta_s:>8} | {t:>6.1f}s")

    print("=" * 100)
    print(f"Total: {ok} OK, {fail} FAILED, {total_t:.0f}s ({total_t/60:.1f} min)")
    print("=" * 100)


if __name__ == "__main__":
    main()
