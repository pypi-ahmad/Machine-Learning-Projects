#!/usr/bin/env python3
"""
Network Intrusion Detection (PyCaret)
=======================================
Binary classification of network connections as normal or attack using the
NSL-KDD dataset. Multi-class attack labels are mapped to a single "attack"
category for binary classification.

Dataset: https://www.kaggle.com/datasets/harivmv/nsl-kdd-dataset
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
KAGGLE_SLUG = "harivmv/nsl-kdd-dataset"

# Standard NSL-KDD column names (41 features + label + difficulty)
NSL_KDD_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]


def get_data(data_dir):
    ds_path = download_kaggle_dataset(KAGGLE_SLUG, data_dir, dataset_name="nsl-kdd-dataset")
    csvs = list(ds_path.glob("*.csv")) + list(data_dir.glob("**/*.csv"))
    txts = list(ds_path.glob("*.txt")) + list(data_dir.glob("**/*.txt"))
    all_files = csvs + txts
    if not all_files:
        raise FileNotFoundError(f"No CSV/TXT found in {ds_path}")

    # Prefer the train file
    train_files = [f for f in all_files if "train" in f.stem.lower()]
    chosen = train_files[0] if train_files else all_files[0]

    # Try reading; add header if missing
    df = pd.read_csv(chosen)
    if len(df.columns) >= 42 and df.columns[0] != "duration":
        df = pd.read_csv(chosen, header=None)
        df.columns = NSL_KDD_COLS[: len(df.columns)]
    logger.info("Loaded %d rows, %d cols from %s", len(df), len(df.columns), chosen.name)
    return df


def preprocess(df):
    df.columns = df.columns.str.strip()

    # Determine target column
    target_col = "label" if "label" in df.columns else df.columns[-1]

    # Drop difficulty level if present
    if "difficulty_level" in df.columns:
        df = df.drop(columns=["difficulty_level"])

    # Map multi-class labels to binary (normal vs attack)
    unique = df[target_col].unique()
    if len(unique) > 2:
        logger.info("Mapping %d unique labels to binary (normal / attack)", len(unique))
        df[target_col] = df[target_col].apply(
            lambda x: "normal" if str(x).strip().lower() == "normal" else "attack"
        )

    df = df.dropna(axis=1, how="all")
    return df


def main():
    args = parse_common_args("Network Intrusion Detection — PyCaret Classification")
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    paths = project_paths(__file__)
    logger.info("=== Network Intrusion Detection (PyCaret) ===")

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
        dataset_missing_metrics(paths["outputs"], "NSL-KDD Intrusion Detection", ["https://www.kaggle.com/datasets/harivmv/nsl-kdd-dataset"])
        return

    df = preprocess(df)

    if args.mode == "smoke":
        df = df.sample(n=min(200, len(df)), random_state=args.seed)
        logger.info("SMOKE TEST: limited to %d rows", len(df))

    target = "label" if "label" in df.columns else df.columns[-1]
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
