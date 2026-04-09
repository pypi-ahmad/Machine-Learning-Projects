#!/usr/bin/env python3
"""
IEEE-CIS Fraud Detection (PyCaret)
====================================
Detect fraud in e-commerce transactions from the IEEE-CIS dataset.
The dataset is large, so it is sampled to 50 000 rows. Uses fix_imbalance=True.

Dataset: https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection
Run:     python run.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import pandas as pd
from shared.utils import (
    download_kaggle_dataset, set_seed, setup_logging, project_paths,
    parse_common_args, configure_cuda_allocator, make_tabular_splits,
    dataset_fingerprint, write_split_manifest,
    save_metrics, run_metadata, dataset_missing_metrics,
    run_tabular_auto,
)

logger = logging.getLogger(__name__)
KAGGLE_SLUG = "lnasiri007/ieeecis-fraud-detection"
MAX_ROWS = 50_000


def get_data(data_dir):
    ds_path = download_kaggle_dataset(KAGGLE_SLUG, data_dir, dataset_name="ieeecis-fraud-detection")
    csvs = list(ds_path.glob("*.csv")) + list(data_dir.glob("**/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {ds_path}")
    # Prefer the transaction file if multiple CSVs exist
    tx_csvs = [c for c in csvs if "transaction" in c.stem.lower() or "train" in c.stem.lower()]
    chosen = tx_csvs[0] if tx_csvs else csvs[0]
    df = pd.read_csv(chosen)
    logger.info("Loaded %d rows, %d cols from %s", len(df), len(df.columns), chosen.name)
    return df


def preprocess(df):
    df.columns = df.columns.str.strip()

    # Drop TransactionID
    if "TransactionID" in df.columns:
        df = df.drop(columns=["TransactionID"])
        logger.info("Dropped TransactionID")

    # Sample if too large
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        logger.info("Sampled dataset to %d rows", MAX_ROWS)

    # Drop columns with > 70 % missing values
    thresh = 0.7 * len(df)
    high_null = [c for c in df.columns if df[c].isna().sum() > thresh]
    if high_null:
        df = df.drop(columns=high_null)
        logger.info("Dropped %d high-null columns", len(high_null))

    df = df.dropna(axis=1, how="all")
    return df


def main():
    args = parse_common_args("IEEE-CIS Fraud Detection — PyCaret Classification")
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    paths = project_paths(__file__)
    logger.info("=== IEEE-CIS Fraud Detection (PyCaret) ===")

    if args.download_only:
        try:
            get_data(paths["data"])
            logger.info("Download complete.")
        except Exception as e:
            logger.error("Download failed: %s", e)
        sys.exit(0)

    try:
        df = get_data(paths["data"])
    except (FileNotFoundError, Exception) as exc:
        logger.error("Dataset error: %s", exc)
        dataset_missing_metrics(paths["outputs"], "IEEE-CIS Fraud Detection", ["https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection"])
        return

    df = preprocess(df)

    if args.mode == "smoke":
        df = df.sample(n=min(200, len(df)), random_state=args.seed)
        logger.info("SMOKE TEST: limited to %d rows", len(df))

    target = "isFraud"
    if target not in df.columns:
        target = df.columns[-1]
        logger.warning("'isFraud' column not found, using last column: %s", target)

    logger.info("Target: %s — classes: %s", target, df[target].value_counts().to_dict())

    # ── Explicit train/val/test split ──
    X_tr, y_tr, X_val, y_val, X_te, y_te = make_tabular_splits(
        df, target, task="classification", seed=args.seed)
    write_split_manifest(
        paths["outputs"],
        dataset_fp=dataset_fingerprint(paths["data"]),
        split_method="stratified_random",
        seed=args.seed,
        counts={"train": len(y_tr), "val": len(y_val), "test": len(y_te)},
    )

    # Auto-ML classification (PyCaret -> LazyPredict -> sklearn)
    metrics = run_tabular_auto(
        output_dir=paths["outputs"], task="classification",
        splits={"X_train": X_tr, "y_train": y_tr,
                "X_val": X_val, "y_val": y_val,
                "X_test": X_te, "y_test": y_te},
    )
    metrics["run_metadata"] = run_metadata(args)
    save_metrics(paths["outputs"], metrics, task_type="classification", mode=args.mode)
    logger.info("Done!")


if __name__ == "__main__":
    main()
