#!/usr/bin/env python3
"""
Project 27 -- Concrete Compressive Strength Prediction

Dataset : Concrete Compressive Strength
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

DATASET = 'Concrete Compressive Strength'
LINKS   = ['https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength']
TARGET  = 'csMPa'

_UCI_URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"



TASK_TYPE = 'tabular'

def get_data() -> pd.DataFrame:
    """Download and load the dataset (UCI ML Repository fallback)."""
    dataset_prompt(DATASET, LINKS)
    csvs = list(DATA_DIR.rglob("*.csv"))
    xls = list(DATA_DIR.rglob("*.xls")) + list(DATA_DIR.rglob("*.xlsx"))
    if not csvs and not xls:
        # Try Kaggle first, fall back to UCI direct download
        try:
            kaggle_download(KAGGLE, DATA_DIR)
        except Exception:
            print("  [INFO] Kaggle unavailable, downloading from UCI...")
            from shared.utils import url_download, extract_archive
            archive = url_download(_UCI_URL, DATA_DIR, filename="concrete.zip")
            extract_archive(archive, DATA_DIR)
        csvs = list(DATA_DIR.rglob("*.csv"))
        xls = list(DATA_DIR.rglob("*.xls")) + list(DATA_DIR.rglob("*.xlsx"))
    if xls:
        df = pd.read_excel(xls[0])
    elif csvs:
        df = pd.read_csv(csvs[0])
    else:
        raise FileNotFoundError("No CSV/XLS found after download")
    target_candidates = [c for c in df.columns if "strength" in c.lower() or "csm" in c.lower()]
    if target_candidates and TARGET not in df.columns:
        df = df.rename(columns={target_candidates[0]: TARGET})
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
