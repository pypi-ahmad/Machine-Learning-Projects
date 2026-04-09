#!/usr/bin/env python3
"""
Project 13 -- COVID-19 Drug Recovery Analysis

Dataset : UNCOVER COVID-19 Challenge
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

DATASET = 'UNCOVER COVID-19 Challenge'
LINKS   = ['https://www.kaggle.com/datasets/roche-data-science-coalition/uncover']
KAGGLE  = 'roche-data-science-coalition/uncover'
TARGET  = 'outcome'



TASK_TYPE = 'tabular'

def get_data() -> pd.DataFrame:
    """Download and load the dataset."""
    dataset_prompt(DATASET, LINKS)
    csvs = list(DATA_DIR.rglob("*.csv"))
    if not csvs:
        kaggle_download(KAGGLE, DATA_DIR)
        csvs = list(DATA_DIR.rglob("*.csv"))
    for csv in csvs:
        try:
            df = pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
        if TARGET in df.columns:
            return df
    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
    for csv in csvs:
        try:
            return pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
    raise FileNotFoundError("No readable CSV found in data directory")


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

    # Truncate early in smoke-test so target detection uses the actual subset
    is_full = (args.mode == 'full')
    if args.smoke_test:
        df = df.head(min(200, len(df)))
        print(f'  [SMOKE] Using first {len(df)} rows only.')

    print(f"  Shape: {df.shape}  |  Target: {TARGET}")

    # Auto-detect target if configured column missing
    _target = TARGET
    if _target not in df.columns:
        # Pick a categorical column with 2-50 unique values (prefer fewer)
        _best, _best_n = None, 999
        for _c in df.columns:
            if df[_c].dtype == "object" and 2 <= df[_c].nunique() <= 50:
                _n = df[_c].nunique()
                if _n < _best_n:
                    _best, _best_n = _c, _n
        _target = _best if _best else df.columns[-1]
        print(f"  [WARN] '{TARGET}' not in columns. Using '{_target}' (n_classes={df[_target].nunique()}).")

    # Drop rows where target is NaN, verify ≥2 classes remain
    df = df.dropna(subset=[_target])
    if df[_target].nunique() < 2:
        print(f"  [ERROR] Target '{_target}' has <2 classes. Cannot classify.")
        return

    print(f"  Classes: {df[_target].nunique()}")

    run_pycaret_classification(df, target=_target, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    from shared.utils import guarded_main
    guarded_main(main, OUTPUT_DIR)
