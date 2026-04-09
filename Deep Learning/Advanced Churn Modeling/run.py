#!/usr/bin/env python3
"""
Project 7 -- Customer Churn Prediction

Dataset : Churn Modelling
Task    : Tabular Classification (PyCaret AutoML)

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
from shared.tabular import run_pycaret_classification

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = 'Churn Modelling'
LINKS   = ['https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling']
KAGGLE  = 'shrutimechlearn/churn-modelling'
TARGET  = 'Exited'



TASK_TYPE = 'tabular'

def get_data() -> pd.DataFrame:
    """Download and load the dataset."""
    dataset_prompt(DATASET, LINKS)
    csv = DATA_DIR / "Churn_Modelling.csv"
    if not csv.exists():
        kaggle_download(KAGGLE, DATA_DIR)
        csvs = list(DATA_DIR.rglob("*.csv"))
        csv = csvs[0] if csvs else csv
    return pd.read_csv(csv)


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
    print(f"  Classes: {df[TARGET].nunique()}")

    is_full = (args.mode == 'full')
    if args.smoke_test:
        df = df.head(min(200, len(df)))
        print(f'  [SMOKE] Using first {len(df)} rows only.')

    run_pycaret_classification(df, target=TARGET, output_dir=OUTPUT_DIR)
    if is_full:
        write_split_manifest(OUTPUT_DIR, dataset_name=DATASET,
            split_counts={'total': len(df)},
            seed=cfg.get('seed', 42))


if __name__ == "__main__":
    from shared.utils import guarded_main
    guarded_main(main, OUTPUT_DIR)
