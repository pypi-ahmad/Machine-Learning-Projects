#!/usr/bin/env python3
"""
Project 43 -- Skin Cancer (HAM10000) Classification

Dataset : Skin Cancer MNIST: HAM10000
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

DATASET = 'Skin Cancer MNIST: HAM10000'
LINKS   = ['https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000']
KAGGLE  = 'kmader/skin-cancer-mnist-ham10000'
MODEL   = 'convnext_tiny.fb_in22k_ft_in1k'
CLASSES = 0  # auto-detect from data directory
EPOCHS  = 20
BATCH   = 32



TASK_TYPE = 'cv'

def get_data():
    """Download and prepare the dataset."""
    dataset_prompt(DATASET, LINKS)
    # Check if data already exists in a usable layout (Train/ or train/)
    for candidate in [DATA_DIR, DATA_DIR / "organized"]:
        subdirs = [d.name.lower() for d in candidate.iterdir() if d.is_dir()] if candidate.exists() else []
        if "train" in subdirs or "test" in subdirs:
            return candidate
    import shutil
    target = DATA_DIR / "organized"
    kaggle_download(KAGGLE, DATA_DIR)
    csv_files = list(DATA_DIR.rglob("*metadata*")) + list(DATA_DIR.rglob("*HAM*.csv"))
    if csv_files:
        import pandas as _pd
        meta = _pd.read_csv(csv_files[0])
        imgs = list(DATA_DIR.rglob("*.jpg"))
        img_map = {p.stem: p for p in imgs}
        import random; random.seed(42)
        meta = meta.sample(frac=1, random_state=42)
        n = len(meta)
        for i, row in enumerate(meta.itertuples()):
            split = "train" if i < .8*n else ("val" if i < .9*n else "test")
            cls = row.dx
            src_img = img_map.get(row.image_id)
            if src_img:
                dest = target / split / cls
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img, dest / src_img.name)
    return target


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
