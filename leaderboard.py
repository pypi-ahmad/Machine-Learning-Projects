"""
PHASE 5b — GLOBAL LEADERBOARD & ANALYTICS

Standalone module that reads artifacts produced by the model governance
layer and generates:
  - Console leaderboard (global ranking + best-per-project + model frequency)
  - artifacts/leaderboard.csv
  - artifacts/leaderboard.png

Works against TWO data sources:
  1. artifacts/global_registry.json  — quick summary per project run
  2. artifacts/<slug>/metrics.json   — full per-project metrics

Usage:
    python leaderboard.py              # full report
    python leaderboard.py --top 5      # show top N only
    python leaderboard.py --no-plot    # skip chart generation
"""

import json
import sys
from pathlib import Path

import pandas as pd

from config import ROOT, ARTIFACTS, REGISTRY_PATH, LEADERBOARD_CSV, LEADERBOARD_PNG


# ====================================================================
# STEP 1 — LOAD REGISTRY
# ====================================================================

def load_registry() -> list[dict]:
    """Load global_registry.json. Returns list of entry dicts."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def load_per_project_metrics() -> list[dict]:
    """Scan artifacts/<slug>/metrics.json for richer metric data."""
    entries = []
    if not ARTIFACTS.exists():
        return entries
    for mf in sorted(ARTIFACTS.glob("*/metrics.json")):
        slug = mf.parent.name
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
        m["project"] = slug
        entries.append(m)
    return entries


def build_dataframe() -> pd.DataFrame:
    """
    Merge registry + per-project metrics into a single DataFrame.

    Registry schema (flat):
        project, best_model, pycaret_model, accuracy, timestamp

    Per-project metrics.json schema (flat):
        best_model_lazypredict, pycaret_model, accuracy, f1, precision,
        recall, lp_accuracy, lp_f1
    """
    registry = load_registry()
    metrics = load_per_project_metrics()

    if not registry and not metrics:
        return pd.DataFrame()

    # Build from per-project metrics (richer)
    if metrics:
        df = pd.DataFrame(metrics)
        # Standardise column names
        rename = {
            "best_model_lazypredict": "best_model_lp",
            "pycaret_model": "model",
            "lp_accuracy": "lp_accuracy",
            "lp_f1": "lp_f1",
        }
        df = df.rename(columns=rename)
    else:
        # Fallback: registry only
        df = pd.DataFrame(registry)
        df = df.rename(columns={
            "best_model": "best_model_lp",
            "pycaret_model": "model",
        })

    # Merge timestamp from registry if available
    if registry and "timestamp" not in df.columns:
        reg_df = pd.DataFrame(registry)[["project", "timestamp"]]
        df = df.merge(reg_df, on="project", how="left")

    # ── STEP 2: Normalise metrics ──
    for col in ("accuracy", "f1", "precision", "recall", "lp_accuracy", "lp_f1"):
        if col not in df.columns:
            df[col] = None
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["accuracy"])
    return df


# ====================================================================
# STEP 3 — GLOBAL RANKING
# ====================================================================

def global_ranking(df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Return leaderboard sorted by accuracy desc."""
    lb = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    lb.index = lb.index + 1  # 1-based rank
    lb.index.name = "rank"
    if top_n:
        lb = lb.head(top_n)
    return lb


# ====================================================================
# STEP 4 — BEST MODEL PER PROJECT
# ====================================================================

def best_per_project(df: pd.DataFrame) -> pd.DataFrame:
    """Best single model per project (by accuracy)."""
    return (
        df.sort_values("accuracy", ascending=False)
        .groupby("project", sort=False)
        .first()
        .sort_values("accuracy", ascending=False)
    )


# ====================================================================
# STEP 5 — MODEL FREQUENCY ANALYSIS
# ====================================================================

def model_frequency(df: pd.DataFrame) -> pd.Series:
    """Count how many projects chose each model type."""
    return df["model"].value_counts()


# ====================================================================
# STEP 6 — SAVE LEADERBOARD CSV
# ====================================================================

def save_leaderboard(lb: pd.DataFrame) -> Path:
    """Save leaderboard to CSV."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lb.to_csv(LEADERBOARD_CSV)
    return LEADERBOARD_CSV


# ====================================================================
# STEP 7 — VISUALIZATION
# ====================================================================

def save_chart(lb: pd.DataFrame, top_n: int = 16) -> Path | None:
    """Bar chart of top models by accuracy."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping chart")
        return None

    chart_data = lb.head(top_n).copy()
    chart_data = chart_data.sort_values("accuracy", ascending=True)  # horizontal bar

    fig, ax = plt.subplots(figsize=(10, max(4, len(chart_data) * 0.5)))

    bars = ax.barh(
        chart_data["project"],
        chart_data["accuracy"],
        color="#4C72B0",
        edgecolor="white",
        height=0.6,
    )

    # Annotate bars
    for bar, model in zip(bars, chart_data["model"]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}  ({model})",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Accuracy")
    ax.set_title("Global Leaderboard — Top Models Across Projects")
    ax.set_xlim(0, min(1.15, chart_data["accuracy"].max() + 0.1))
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(LEADERBOARD_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return LEADERBOARD_PNG


# ====================================================================
# STEP 8 — CLI ENTRYPOINT
# ====================================================================

def print_section(title: str):
    """Print a formatted section header."""
    w = 60
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


def main():
    # Parse simple CLI args
    args = set(sys.argv[1:])
    top_n = None
    no_plot = "--no-plot" in args
    if "--top" in args:
        try:
            idx = sys.argv.index("--top")
            top_n = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            top_n = 10

    # ── Load ──
    print_section("GLOBAL LEADERBOARD")
    df = build_dataframe()

    if df.empty:
        print("\n  No data found.")
        print("  Run the notebooks first to populate artifacts/global_registry.json")
        print("  and artifacts/<project>/metrics.json.\n")

        # Generate empty placeholder files so downstream tools don't break
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "project", "model", "accuracy", "f1", "precision", "recall"
        ]).to_csv(LEADERBOARD_CSV, index=False)
        print(f"  Created empty {LEADERBOARD_CSV.relative_to(ROOT)}")
        return

    print(f"\n  Loaded {len(df)} entries from {ARTIFACTS.relative_to(ROOT)}/")

    # ── Global Ranking ──
    lb = global_ranking(df, top_n=top_n)
    cols = ["project", "model", "accuracy", "f1", "precision", "recall"]
    cols = [c for c in cols if c in lb.columns]

    print_section("RANKING" + (f" (top {top_n})" if top_n else ""))
    print(lb[cols].to_string())

    # ── Best per project ──
    bpp = best_per_project(df)
    bpp_cols = [c for c in ["model", "accuracy", "f1"] if c in bpp.columns]

    print_section("BEST MODEL PER PROJECT")
    print(bpp[bpp_cols].to_string())

    # ── Model frequency ──
    mf = model_frequency(df)

    print_section("MODEL FREQUENCY")
    print(mf.to_string())

    # ── Save CSV ──
    csv_path = save_leaderboard(lb)
    print(f"\n  Leaderboard saved to {csv_path.relative_to(ROOT)}")

    # ── Chart ──
    if not no_plot:
        png_path = save_chart(lb)
        if png_path:
            print(f"  Chart saved to {png_path.relative_to(ROOT)}")

    # ── Done ──
    print_section("DONE")
    print(f"  Projects ranked: {len(df)}")
    print(f"  Unique models:   {df['model'].nunique()}")
    if "accuracy" in df.columns:
        print(f"  Accuracy range:  {df['accuracy'].min():.4f} – {df['accuracy'].max():.4f}")
    print()


if __name__ == "__main__":
    main()
