#!/usr/bin/env python3
"""
Insurance Fraud Detection (PyCaret)
=====================================
Classify insurance claims as fraudulent or legitimate.

Dataset: https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection
Run:     python run.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import pandas as pd
from shared.utils import (
    download_kaggle_dataset, set_seed, setup_logging, project_paths,
    run_tabular_auto,
    parse_common_args, configure_cuda_allocator, make_tabular_splits,
    dataset_fingerprint, write_split_manifest,
    save_metrics, run_metadata, dataset_missing_metrics,
)

logger = logging.getLogger(__name__)
KAGGLE_SLUG = "arpan129/insurance-fraud-detection"


def get_data(data_dir):
    ds_path = download_kaggle_dataset(KAGGLE_SLUG, data_dir, dataset_name="insurance-fraud-detection")
    csvs = list(ds_path.glob("*.csv")) + list(data_dir.glob("**/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {ds_path}")
    df = pd.read_csv(csvs[0])
    logger.info("Loaded %d rows, %d cols from %s", len(df), len(df.columns), csvs[0].name)
    return df


def preprocess(df):
    df.columns = df.columns.str.strip()

    # Drop ID-like columns
    id_cols = [c for c in df.columns if c.lower() in (
        "policy_number", "policynumber", "id", "claim_id", "claimid",
        "insured_zip", "incident_location",
    )]
    if id_cols:
        df = df.drop(columns=id_cols)
        logger.info("Dropped ID-like columns: %s", id_cols)

    # Replace '?' with NaN (common in this dataset)
    df = df.replace("?", pd.NA)

    df = df.dropna(axis=1, how="all")
    return df


def _detect_target(df):
    """Return the fraud-target column name."""
    candidates = ["fraud_reported", "FRAUD", "Fraud", "fraud", "is_fraud", "Class"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[-1]


def main():
    args = parse_common_args("Insurance Fraud Detection — PyCaret Classification")
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    paths = project_paths(__file__)
    logger.info("=== Insurance Fraud Detection (PyCaret) ===")

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
        dataset_missing_metrics(paths["outputs"], "Insurance Fraud Detection", ["https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection"])
        return

    df = preprocess(df)

    if args.mode == "smoke":
        df = df.sample(n=min(200, len(df)), random_state=args.seed)
        logger.info("SMOKE TEST: limited to %d rows", len(df))

    target = _detect_target(df)
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
