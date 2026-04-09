#!/usr/bin/env python3
"""TV Show Recommendation — Content-Based (PyCaret)

Dataset : https://www.kaggle.com/datasets/riteshswami08/10000-popular-tv-shows-dataset-tmdb
Approach: TMDB metadata (no user-level ratings).  Use content-based PyCaret
          regression to predict vote_average from features like vote_count,
          popularity, number_of_seasons, etc.
          Saves model comparison table + regression report to outputs/.
Run     : python "Recommendation Systems/11 TV Show Recommendation System/run.py"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import numpy as np
import pandas as pd

from shared.utils import (
    download_kaggle_dataset,
    set_seed,
    setup_logging,
    project_paths,
    get_device,
    save_regression_report,
    ensure_dir,
    parse_common_args,
    save_metrics,
    dataset_missing_metrics,
    run_tabular_auto,
    configure_cuda_allocator,
    run_metadata,
    dataset_fingerprint,
    write_split_manifest,
    make_tabular_splits,
)

logger = logging.getLogger(__name__)

KAGGLE_SLUG = "riteshswami08/10000-popular-tv-shows-dataset-tmdb"
SAMPLE_LIMIT = 50_000

# Candidate target columns (prefer vote_average)
TARGET_COLS = [
    "vote_average", "voteaverage", "rating", "score",
    "imdb_rating", "imdb_score",
]

# Numeric feature columns typically in TMDB data
NUMERIC_FEATURES = [
    "vote_count", "votecount", "popularity", "number_of_seasons",
    "number_of_episodes", "episode_run_time", "revenue", "budget",
    "runtime",
]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _load_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.rglob("*.csv")) + sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CSV/parquet files under {data_dir}")

    # Pick largest CSV (most likely the main dataset)
    best_file = max(files, key=lambda f: f.stat().st_size)
    logger.info("Loading %s (%d KB)", best_file.name, best_file.stat().st_size // 1024)
    if best_file.suffix == ".csv":
        return pd.read_csv(best_file)
    return pd.read_parquet(best_file)


def _select_features(df: pd.DataFrame, target_col: str) -> list[str]:
    """Select numeric feature columns, excluding the target."""
    # Start with known TMDB feature names
    features = []
    for cand in NUMERIC_FEATURES:
        col = _find_col(df, [cand])
        if col and col != target_col:
            features.append(col)

    # Also include any other numeric columns not in the feature list
    for c in df.select_dtypes(include=[np.number]).columns:
        if c != target_col and c not in features:
            # Skip ID-like columns
            if "id" in c.lower() or df[c].nunique() == len(df):
                continue
            features.append(c)

    return features


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global SAMPLE_LIMIT

    args = parse_common_args("TV Show Recommendation — PyCaret")
    paths = project_paths(__file__)
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    _ = get_device()  # log device info

    if args.mode == "smoke":
        SAMPLE_LIMIT = 1000

    # Data
    data_dir = download_kaggle_dataset(KAGGLE_SLUG, paths["data"])
    if args.download_only:
        logger.info("Download complete — exiting (--download-only).")
        sys.exit(0)

    try:
        df = _load_data(data_dir)
    except Exception as exc:
        logger.warning("Dataset loading failed: %s", exc)
        dataset_missing_metrics(
            paths["outputs"], KAGGLE_SLUG,
            [f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}"],
        )
        return

    logger.info("Loaded dataframe: %s rows, %d cols", f"{len(df):,}", df.shape[1])
    logger.info("Columns: %s", list(df.columns))

    out_dir = ensure_dir(paths["outputs"])

    # ── Identify target ──────────────────────────────────────────────────
    target_col = _find_col(df, TARGET_COLS)
    if target_col is None:
        # Fall back to first numeric column with reasonable range
        for c in df.select_dtypes(include=[np.number]).columns:
            if 2 < df[c].nunique() < len(df) * 0.9:
                target_col = c
                break
    if target_col is None:
        raise ValueError("No suitable target column found.")
    logger.info("Target column: %s", target_col)

    # ── Select features ──────────────────────────────────────────────────
    features = _select_features(df, target_col)
    if not features:
        raise ValueError("No numeric features found for regression.")
    logger.info("Features (%d): %s", len(features), features)

    data = df[features + [target_col]].dropna()
    if len(data) > SAMPLE_LIMIT:
        data = data.sample(SAMPLE_LIMIT, random_state=args.seed)
    logger.info("Training data: %d rows", len(data))

    # Dataset fingerprint
    ds_fp = dataset_fingerprint(data_dir)

    # ── Proper train/val/test splits ─────────────────────────────────────
    X_tr, y_tr, X_va, y_va, X_te, y_te = make_tabular_splits(
        data, target_col, task="regression", seed=args.seed,
    )
    splits = dict(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te)
    write_split_manifest(
        out_dir,
        dataset_fp=ds_fp,
        split_method="tabular_random",
        seed=args.seed,
        counts={"train": len(y_tr), "val": len(y_va), "test": len(y_te)},
    )

    # ── Auto-ML regression (PyCaret -> LazyPredict -> sklearn) ────────
    metrics = run_tabular_auto(data, target=target_col, output_dir=out_dir, task="regression", session_id=args.seed, splits=splits)

    # Final metrics
    metrics["run_metadata"] = run_metadata(args)
    metrics["n_test"] = len(y_te)
    metrics["split"] = "test"
    save_metrics(out_dir, metrics, task_type="recsys", mode=args.mode)

    # ── Scatter plot ─────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("TV Show Recommendation — Metrics Summary")
        ax.text(0.5, 0.5, "\n".join(f"{k}: {v}" for k, v in metrics.items() if isinstance(v, (int, float))),
                transform=ax.transAxes, ha="center", va="center", fontsize=10, family="monospace")
        fig.tight_layout()
        fig.savefig(out_dir / "scatter.png", dpi=150)
        plt.close(fig)
        logger.info("Scatter plot saved.")
    except ImportError:
        logger.warning("matplotlib not available — skipping scatter plot.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
