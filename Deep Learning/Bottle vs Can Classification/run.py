#!/usr/bin/env python3
"""
Project 42 -- Bottle vs Can Classification

Dataset : Bottles and Cans
Model   : convnext_tiny.fb_in22k_ft_in1k (timm)
Task    : Image Classification

Usage:
    python run.py                # full training
    python run.py --smoke-test   # quick sanity check (1 epoch, 2 batches)
    python run.py --download-only
    python run.py --epochs 5 --batch-size 64 --device cuda
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.utils import (
    seed_everything, get_device, dataset_prompt, kaggle_download, ensure_dir,
    parse_common_args, load_profile, resolve_config, write_split_manifest,
    EarlyStopping, run_with_oom_backoff)
from shared.cv import (create_dataloaders, build_timm_model,
                        train_model, evaluate_model)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = 'Bottles and Cans'
LINKS   = ['https://www.kaggle.com/datasets/trolukovich/bottles-and-cans']
KAGGLE  = 'trolukovich/bottles-and-cans'
MODEL   = 'convnext_tiny.fb_in22k_ft_in1k'
CLASSES = 0
EPOCHS  = 15
BATCH   = 32



TASK_TYPE = 'cv'

def get_data():
    """Download and prepare the dataset."""
    dataset_prompt(DATASET, LINKS)
    if not list(DATA_DIR.rglob("*.jpg")) and not list(DATA_DIR.rglob("*.png")):
        kaggle_download(KAGGLE, DATA_DIR)
    for child in sorted(DATA_DIR.iterdir()):
        if child.is_dir() and child.name != "__MACOSX":
            return child
    return DATA_DIR


def main():
    args = parse_common_args()
    profile = load_profile(args.profile)
    cfg = resolve_config(args, profile, TASK_TYPE)
    seed_everything(cfg.get('seed', 42))
    device = get_device(args.device)
    ensure_dir(OUTPUT_DIR)

    data_root = get_data()
    if args.download_only:
        print("  Dataset ready. Exiting (--download-only).")
        return

    is_full    = (args.mode == 'full')
    epochs     = args.epochs or cfg.get('epochs', EPOCHS)
    batch_size = args.batch_size or cfg.get('batch_size', BATCH)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get('num_workers', 2)
    use_amp    = not args.no_amp and cfg.get('amp', True)
    img_size   = cfg.get('img_size', 224)
    max_batches = 2 if args.smoke_test else None
    grad_accum  = cfg.get('grad_accum_steps', 1) if is_full else 1
    freeze_bb   = cfg.get('freeze_backbone_epochs', 0) if is_full else 0
    es_on       = cfg.get('early_stopping', False) if is_full else False
    es_patience = cfg.get('patience', 3)
    if args.smoke_test:
        epochs = 1

    train_dl, val_dl, test_dl, class_names = create_dataloaders(
        data_root, img_size=img_size, batch_size=batch_size,
        num_workers=num_workers,
    )
    model = build_timm_model(MODEL, num_classes=CLASSES or len(class_names))
    model = train_model(
        model, train_dl, val_dl,
        epochs=epochs, lr=cfg.get('lr', 1e-4), device=device, output_dir=OUTPUT_DIR,
        use_amp=use_amp, max_batches=max_batches,
        grad_accum_steps=grad_accum,
        early_stopping=es_on, patience=es_patience,
        freeze_backbone_epochs=freeze_bb,
    )
    evaluate_model(model, test_dl, class_names, device=device,
                   output_dir=OUTPUT_DIR, max_batches=max_batches)


    if is_full:
        write_split_manifest(OUTPUT_DIR, dataset_name=DATASET,
            split_counts={'train': len(train_dl.dataset),
                          'val': len(val_dl.dataset),
                          'test': len(test_dl.dataset)},
            seed=cfg.get('seed', 42))
if __name__ == "__main__":
    from shared.utils import guarded_main
    guarded_main(main, OUTPUT_DIR)
