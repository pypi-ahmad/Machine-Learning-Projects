#!/usr/bin/env python3
"""
Weather Chatbot — Weather Classification
==========================================
Classifies historical weather data into temperature-range categories
(cold / mild / warm / hot) using PyCaret's automated ML pipeline.

The dataset contains hourly weather measurements for multiple cities.
We engineer categorical targets from temperature data and train a
traditional ML classifier via PyCaret.

Dataset: https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data
Run:     python run.py
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
    run_tabular_auto,
    parse_common_args,
    save_metrics,
    dataset_missing_metrics,
    configure_cuda_allocator,
    make_tabular_splits,
    write_split_manifest,
    dataset_fingerprint,
    run_metadata,
)

logger = logging.getLogger(__name__)

KAGGLE_SLUG = "selfishgene/historical-hourly-weather-data"
SEED = 42
MAX_ROWS = 30_000

# Temperature bins (Kelvin thresholds, typical for this dataset)
# cold < 270 K (~-3 °C), mild 270–290 K, warm 290–305 K, hot > 305 K
TEMP_BINS_K = [0, 270, 290, 305, 400]
TEMP_LABELS = ["cold", "mild", "warm", "hot"]

# Celsius fallback
TEMP_BINS_C = [-60, -3, 17, 32, 60]


# ═════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════
def _load_temperature_csv(data_dir: Path) -> pd.DataFrame:
    """Find and load the temperature CSV from the dataset."""
    ds_path = download_kaggle_dataset(
        KAGGLE_SLUG, data_dir,
        dataset_name="Historical Hourly Weather Data",
    )

    # Look for temperature file specifically, then any CSV
    search_dirs = [ds_path, data_dir]
    temp_file = None
    for d in search_dirs:
        for pattern in ["*emperature*", "*temp*"]:
            matches = sorted(d.glob(pattern))
            if matches:
                temp_file = matches[0]
                break
        if temp_file:
            break

    if temp_file is None:
        # Fallback: pick the largest CSV
        csvs = sorted(ds_path.glob("*.csv")) + sorted(data_dir.glob("**/*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {ds_path}")
        temp_file = max(csvs, key=lambda f: f.stat().st_size)

    logger.info("Loading temperature data from %s", temp_file.name)
    df = pd.read_csv(temp_file)
    return df


def _try_load_description(data_dir: Path):
    """Try to load weather_description.csv for richer features."""
    for d in [data_dir, data_dir / "historical-hourly-weather-data"]:
        for pattern in ["*description*", "*weather_desc*"]:
            matches = sorted(d.glob(pattern))
            if matches:
                return pd.read_csv(matches[0])
    return None


def get_data(data_dir: Path) -> pd.DataFrame:
    """Build a classification-ready DataFrame from the weather dataset."""
    df_temp = _load_temperature_csv(data_dir)

    # The dataset has a 'datetime' column and one column per city
    if "datetime" in df_temp.columns:
        df_temp = df_temp.drop(columns=["datetime"], errors="ignore")

    # Melt to long format: city + temperature
    df_long = df_temp.melt(var_name="city", value_name="temperature").dropna()
    logger.info("Melted temperature data: %d rows", len(df_long))

    # Detect Kelvin vs Celsius (Kelvin values typically > 200)
    median_temp = df_long["temperature"].median()
    if median_temp > 200:
        bins, labels = TEMP_BINS_K, TEMP_LABELS
        logger.info("Detected Kelvin scale (median=%.1f)", median_temp)
    else:
        bins, labels = TEMP_BINS_C, TEMP_LABELS
        logger.info("Detected Celsius scale (median=%.1f)", median_temp)

    df_long["temp_class"] = pd.cut(
        df_long["temperature"], bins=bins, labels=labels, include_lowest=True,
    )
    df_long = df_long.dropna(subset=["temp_class"]).reset_index(drop=True)

    # Try to load description for extra features
    desc_df = _try_load_description(data_dir)
    if desc_df is not None and "datetime" not in desc_df.columns:
        desc_df = None
    # Description merging is optional — skip if format differs

    # Add engineered features
    df_long["temp_zscore"] = (
        (df_long["temperature"] - df_long["temperature"].mean())
        / df_long["temperature"].std()
    )

    # Sample to keep things manageable
    if len(df_long) > MAX_ROWS:
        df_long = df_long.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
        logger.info("Sampled down to %d rows", MAX_ROWS)

    # Encode city as numeric
    df_long["city_code"] = df_long["city"].astype("category").cat.codes

    # Final columns for modelling
    model_df = df_long[["temperature", "temp_zscore", "city_code", "temp_class"]].copy()
    model_df["temp_class"] = model_df["temp_class"].astype(str)
    logger.info("Class distribution:\n%s", model_df["temp_class"].value_counts().to_string())
    return model_df


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════
def main():
    args = parse_common_args("Weather Chatbot — Weather Classification")
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    paths = project_paths(__file__)

    # -- download-only gate
    if args.download_only:
        try:
            get_data(paths["data"])
            logger.info("Download complete.")
        except Exception as e:
            logger.error("Download failed: %s", e)
        sys.exit(0)

    # 1. Build dataset
    try:
        df = get_data(paths["data"])
    except (FileNotFoundError, Exception) as exc:
        logger.error("Dataset error: %s", exc)
        dataset_missing_metrics(
            paths["outputs"],
            "Historical Hourly Weather Data",
            ["https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data"],
        )
        return

    if args.mode == "smoke":
        df = df.sample(n=min(200, len(df)), random_state=args.seed)
        logger.info("SMOKE TEST: %d rows", len(df))

    # 2. Split data
    X_tr, y_tr, X_v, y_v, X_te, y_te = make_tabular_splits(df, "temp_class", "classification", args.seed)
    splits = {"X_train": X_tr, "y_train": y_tr, "X_val": X_v, "y_val": y_v, "X_test": X_te, "y_test": y_te}
    write_split_manifest(
        paths["outputs"],
        dataset_fp=dataset_fingerprint(paths["data"]),
        split_method="stratified_random",
        seed=args.seed,
        counts={"train": len(y_tr), "val": len(y_v), "test": len(y_te)},
    )

    # 3. Run auto-ML classification (PyCaret -> LazyPredict -> sklearn)
    logger.info("Running tabular auto-ML classification pipeline …")
    metrics = run_tabular_auto(df, target="temp_class", output_dir=paths["outputs"],
                               task="classification", session_id=args.seed, splits=splits)
    metrics["run_metadata"] = run_metadata(args)
    save_metrics(paths["outputs"], metrics, task_type="classification", mode=args.mode)

    logger.info("Done.")


if __name__ == "__main__":
    main()
