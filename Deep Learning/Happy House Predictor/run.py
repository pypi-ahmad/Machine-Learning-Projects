#!/usr/bin/env python3
"""
Project 15 -- Housing Price Prediction

Dataset : California Housing (sklearn built-in, replaces deprecated Boston Housing)
Task    : Tabular Regression (PyCaret AutoML)

Usage:
    python run.py                # full training
    python run.py --smoke-test   # quick sanity check
    python run.py --download-only
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from shared.utils import (
    seed_everything, dataset_prompt, kaggle_download, ensure_dir, parse_common_args,
    save_metrics, load_profile, resolve_config, write_split_manifest,
    make_tabular_splits)
from shared.tabular import run_pycaret_regression

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = 'California Housing (sklearn)'
LINKS   = ['https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset']
TARGET  = 'MedHouseVal'



TASK_TYPE = 'tabular'

def get_data() -> pd.DataFrame:
    """Load the California Housing dataset from sklearn (always available)."""
    dataset_prompt(DATASET, LINKS)
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df


def main():
    args = parse_common_args()
    profile = load_profile(args.profile)
    cfg = resolve_config(args, profile, TASK_TYPE)
    seed_everything(cfg.get('seed', 42))
    ensure_dir(OUTPUT_DIR)

    df = get_data()
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    print(f"  Shape: {df.shape}  |  Target: {TARGET}")

    is_full = (args.mode == 'full')
    if args.smoke_test:
        df = df.head(min(200, len(df)))
        print(f'  [SMOKE] Using first {len(df)} rows only.')

    run_pycaret_regression(df, target=TARGET, output_dir=OUTPUT_DIR)
    if is_full:
        write_split_manifest(OUTPUT_DIR, dataset_name=DATASET,
            split_counts={'total': len(df)},
            seed=cfg.get('seed', 42))


if __name__ == "__main__":
    from shared.utils import guarded_main
    guarded_main(main, OUTPUT_DIR)
