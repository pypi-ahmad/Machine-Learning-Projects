#!/usr/bin/env python3
"""
Project 49 -- IMDB Sentiment Analysis

Dataset : IMDB 50K Movie Reviews
Model   : distilbert-base-uncased (HuggingFace)
Task    : Text Classification

Usage:
    python run.py                # full training
    python run.py --smoke-test   # quick sanity check (4 train steps)
    python run.py --download-only
    python run.py --epochs 5 --batch-size 32 --device cuda
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from shared.utils import (
    seed_everything, get_device, dataset_prompt, kaggle_download, ensure_dir,
    parse_common_args, load_profile, resolve_config, write_split_manifest,
    make_tabular_splits)
from shared.nlp import (build_hf_classifier, tokenize_texts,
                         train_hf_classifier, evaluate_hf_classifier)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = 'IMDB 50K Movie Reviews'
LINKS   = ['https://ai.stanford.edu/~amaas/data/sentiment/', 'https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews']
KAGGLE  = 'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews'
HF_MODEL = 'distilbert-base-uncased'
NUM_LABELS = 2
EPOCHS = 3
BATCH  = 16



TASK_TYPE = 'nlp'

def get_data():
    """Download, load and split text data.

    Returns (train_texts, train_labels, test_texts, test_labels, class_names).
    """
    dataset_prompt(DATASET, LINKS)
    csvs = list(DATA_DIR.rglob("*.csv"))
    if not csvs:
        kaggle_download(KAGGLE, DATA_DIR)
        csvs = list(DATA_DIR.rglob("*.csv"))
    df = pd.read_csv(csvs[0])
    lab_map = {"positive": 1, "negative": 0}
    df["_label"] = df["sentiment"].map(lab_map)
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])
    return (tr["review"].tolist(), tr["_label"].tolist(),
            te["review"].tolist(), te["_label"].tolist(),
            ["negative", "positive"])


def main():
    args = parse_common_args()
    profile = load_profile(args.profile)
    cfg = resolve_config(args, profile, TASK_TYPE)
    seed_everything(cfg.get('seed', 42))
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    train_texts, train_labels, test_texts, test_labels, class_names = get_data()
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    print(f"  Train: {len(train_texts)}  |  Test: {len(test_texts)}  "
          f"|  Classes: {class_names}")

    is_full    = (args.mode == 'full')
    epochs     = args.epochs or cfg.get('epochs', EPOCHS)
    batch_size = args.batch_size or cfg.get('batch_size', BATCH)
    use_amp    = not args.no_amp and cfg.get('amp', True)
    max_steps  = 4 if args.smoke_test else -1
    grad_accum = cfg.get('grad_accum_steps', 1) if is_full else 1
    gc_on      = cfg.get('gradient_checkpointing', False) if is_full else False
    es_patience = cfg.get('patience', 2) if is_full else None

    if args.smoke_test:
        epochs = 1

    model, tokenizer = build_hf_classifier(HF_MODEL, num_labels=NUM_LABELS or len(class_names))
    train_ds = tokenize_texts(train_texts, train_labels, tokenizer)
    test_ds  = tokenize_texts(test_texts, test_labels, tokenizer)

    trainer = train_hf_classifier(
        model, train_ds, test_ds, OUTPUT_DIR,
        epochs=epochs, batch_size=batch_size, lr=cfg.get('lr', 2e-5),
        use_amp=use_amp, max_steps=max_steps,
        gradient_checkpointing=gc_on,
        early_stopping_patience=es_patience,
        grad_accum_steps=grad_accum,
    )
    evaluate_hf_classifier(trainer, test_ds, class_names, OUTPUT_DIR)
    if is_full:
        write_split_manifest(OUTPUT_DIR, dataset_name=DATASET,
            split_counts={'train': len(train_ds), 'test': len(test_ds)},
            seed=cfg.get('seed', 42))


if __name__ == "__main__":
    from shared.utils import guarded_main
    guarded_main(main, OUTPUT_DIR)
