#!/usr/bin/env python3
"""
Anomaly Detection in Social Networks — Twitter Bot Detection
=============================================================
PyCaret classification on twitter bot detection dataset.

Dataset: https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset
Run:     python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import pandas as pd
from shared.utils import (
    download_kaggle_dataset,
    set_seed,
    setup_logging,
    project_paths,
    run_tabular_auto,
    save_classification_report,
    parse_common_args,
    configure_cuda_allocator,
    make_tabular_splits,
    dataset_fingerprint,
    write_split_manifest,
    save_metrics,
    run_metadata,
    dataset_missing_metrics,
)

logger = logging.getLogger(__name__)

KAGGLE_SLUG = "goyaladi/twitter-bot-detection-dataset"


def get_data(data_dir: Path) -> pd.DataFrame:
    ds_path = download_kaggle_dataset(KAGGLE_SLUG, data_dir, dataset_name="Twitter Bot Detection")
    # Find the CSV
    csvs = list(ds_path.glob("*.csv")) + list(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {ds_path}")
    df = pd.read_csv(csvs[0])
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), csvs[0].name)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop ID-like columns and high-cardinality text
    drop_cols = [c for c in df.columns if c.lower() in (
        "id", "id_str", "screen_name", "name", "description",
        "url", "profile_image_url", "profile_banner_url",
        "profile_background_image_url", "created_at",
    )]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # Convert boolean columns
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)
    # Drop columns with >50% missing
    df = df.dropna(thresh=len(df) * 0.5, axis=1)
    df = df.dropna()
    return df


def main():
    args = parse_common_args("Twitter Bot Detection — PyCaret Classification")
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    paths = project_paths(__file__)

    logger.info("=== Twitter Bot Detection (PyCaret) ===")

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
        dataset_missing_metrics(paths["outputs"], "Twitter Bot Detection", ["https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset"])
        return

    df = preprocess(df)

    if args.mode == "smoke":
        df = df.sample(n=min(200, len(df)), random_state=args.seed)
        logger.info("SMOKE TEST: limited to %d rows", len(df))

    # Identify target column (likely 'bot' or 'label' or 'is_bot')
    target_candidates = ["bot", "label", "is_bot", "Bot", "Label", "target"]
    target = None
    for c in target_candidates:
        if c in df.columns:
            target = c
            break
    if target is None:
        # Use last column as target
        target = df.columns[-1]
    logger.info("Target column: %s (classes: %s)", target, df[target].value_counts().to_dict())

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
