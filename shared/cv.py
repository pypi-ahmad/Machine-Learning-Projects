"""
Shared Computer Vision utilities — timm models, dataloaders, training, evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import (ensure_dir, plot_confusion_matrix, plot_training_curves,
                    save_metrics, EarlyStopping)


# ═══════════════════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════════════════

def get_transforms(img_size: int = 224, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-detect directory layout
# ═══════════════════════════════════════════════════════════════════════════════

_TRAIN_NAMES = ["train", "Train", "training_set", "training", "TRAIN"]
_VAL_NAMES   = ["val", "Val", "validation", "Validation", "valid", "dev"]
_TEST_NAMES  = ["test", "Test", "test_set", "testing", "TEST"]


def _find_subdir(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = root / name
        if p.is_dir():
            # handle nested duplicates  e.g. training_set/training_set
            inner = p / name
            return inner if inner.is_dir() else p
    return None


def auto_find_splits(data_dir: str | Path):
    """Return (train_dir, val_dir, test_dir) or raise."""
    data_dir = Path(data_dir)
    train = _find_subdir(data_dir, _TRAIN_NAMES)
    val   = _find_subdir(data_dir, _VAL_NAMES)
    test  = _find_subdir(data_dir, _TEST_NAMES)

    if train is None:
        # Maybe the root itself is the dataset (single-level with class dirs)
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if subdirs:
            train = data_dir
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════════
# Dataloaders
# ═══════════════════════════════════════════════════════════════════════════════

def create_dataloaders(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    val_frac: float = 0.15,
    train_dir: str | None = None,
    val_dir: str | None = None,
    test_dir: str | None = None,
):
    """
    Build train/val/test DataLoaders from an ImageFolder layout.
    Auto-detects split directories if not provided.
    Returns (train_dl, val_dl, test_dl, class_names).
    """
    data_dir = Path(data_dir)
    train_tf = get_transforms(img_size, is_train=True)
    eval_tf  = get_transforms(img_size, is_train=False)

    # --- resolve paths ---
    if train_dir is not None:
        t_path = data_dir / train_dir
        v_path = (data_dir / val_dir) if val_dir else None
        te_path = (data_dir / test_dir) if test_dir else None
    else:
        t_path, v_path, te_path = auto_find_splits(data_dir)

    if t_path is None or not t_path.exists():
        raise FileNotFoundError(
            f"Cannot find training data under {data_dir}. "
            "Provide train_dir= or organise data into train/val/test folders."
        )

    # Remove junk dirs that confuse ImageFolder (__pycache__, __MACOSX)
    import shutil as _shutil
    _junk_names = ("__pycache__", "__MACOSX")
    # Clean from data root recursively AND from each split path
    for _root in [data_dir, t_path, v_path, te_path]:
        if _root and _root.is_dir():
            for _junk in _root.rglob("__MACOSX"):
                if _junk.is_dir():
                    _shutil.rmtree(_junk)
            for _junk in _root.rglob("__pycache__"):
                if _junk.is_dir():
                    _shutil.rmtree(_junk)

    train_ds = datasets.ImageFolder(t_path, transform=train_tf)
    class_names = train_ds.classes
    print(f"  Classes ({len(class_names)}): {class_names[:10]}{'...' if len(class_names) > 10 else ''}")

    # --- validation ---
    if v_path and v_path.exists():
        val_ds = datasets.ImageFolder(v_path, transform=eval_tf)
    else:
        n_val   = int(val_frac * len(train_ds))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        # random_split doesn't change transforms; wrap val portion
        # (since both come from the same ImageFolder, train augmentation is shared)

    # --- test ---
    if te_path and te_path.exists():
        test_ds = datasets.ImageFolder(te_path, transform=eval_tf)
    else:
        test_ds = val_ds  # fallback: use val as test

    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_dl, val_dl, test_dl, class_names


# ═══════════════════════════════════════════════════════════════════════════════
# Model builder (timm)
# ═══════════════════════════════════════════════════════════════════════════════

def build_timm_model(model_name: str, num_classes: int, pretrained: bool = True):
    import timm
    model = timm.create_model(model_name, pretrained=pretrained,
                              num_classes=num_classes)
    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model : {model_name}")
    print(f"  Params: {n_params:,} total | {n_trainable:,} trainable")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    *,
    epochs: int = 20,
    lr: float = 1e-4,
    device: torch.device | str = "cuda",
    output_dir: str | Path = "outputs",
    use_amp: bool = True,
    max_batches: int | None = None,
    grad_accum_steps: int = 1,
    early_stopping: bool = False,
    patience: int = 3,
    freeze_backbone_epochs: int = 0,
) -> nn.Module:
    output_dir = ensure_dir(output_dir)
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)

    # ── Backbone freezing (transfer-learning VRAM saver) ──────────────────
    _frozen_params: list[nn.Parameter] = []
    if freeze_backbone_epochs > 0:
        # Freeze everything except the classifier head
        for name, param in model.named_parameters():
            is_head = any(k in name for k in ("head", "classifier", "fc", "last_linear"))
            if not is_head:
                param.requires_grad = False
                _frozen_params.append(param)
        n_frozen = sum(p.numel() for p in _frozen_params)
        print(f"  [FREEZE] Backbone frozen for first {freeze_backbone_epochs} epochs "
              f"({n_frozen:,} params)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    amp_on  = use_amp and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=amp_on)

    best_val_acc = -1.0
    es = EarlyStopping(patience=patience, mode="max") if early_stopping else None
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        # ── Unfreeze backbone after N epochs ──────────────────────────
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            for param in _frozen_params:
                param.requires_grad = True
            # Rebuild optimizer to include all params
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1,
                                          weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch + 1)
            print(f"  [UNFREEZE] Backbone unfrozen at epoch {epoch} (lr={lr * 0.1:.1e})")

        # ── train ─────────────────────────────────────────────
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss = run_correct = run_total = 0
        for bi, (X, y) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} train", leave=False)):
            if max_batches and bi >= max_batches:
                break
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_on):
                logits = model(X)
                loss = criterion(logits, y)
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

            if (bi + 1) % grad_accum_steps == 0 or (bi + 1) == len(train_dl):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            run_loss += loss.item() * grad_accum_steps * X.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            run_total += y.size(0)

        t_loss = run_loss / max(run_total, 1)
        t_acc  = run_correct / max(run_total, 1)

        # ── validate ──────────────────────────────────────────
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for bi, (X, y) in enumerate(tqdm(val_dl, desc=f"Epoch {epoch}/{epochs} val", leave=False)):
                if max_batches and bi >= max_batches:
                    break
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp_on):
                    logits = model(X)
                    loss = criterion(logits, y)
                v_loss += loss.item() * X.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += y.size(0)

        v_loss /= max(v_total, 1)
        v_acc   = v_correct / max(v_total, 1)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"  Epoch {epoch:3d}  "
              f"t_loss={t_loss:.4f}  t_acc={t_acc:.4f}  "
              f"v_loss={v_loss:.4f}  v_acc={v_acc:.4f}")

        if v_acc > best_val_acc or epoch == 1:
            best_val_acc = v_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        if es and es.step(v_acc):
            break

    plot_training_curves(history, output_dir)
    ckpt = output_dir / "best_model.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: nn.Module,
    test_dl: DataLoader,
    class_names: list[str],
    *,
    device: torch.device | str = "cuda",
    output_dir: str | Path = "outputs",
    max_batches: int | None = None,
) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report,
                                  f1_score)
    output_dir = ensure_dir(output_dir)
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for bi, (X, y) in enumerate(tqdm(test_dl, desc="Evaluating")):
            if max_batches and bi >= max_batches:
                break
            X = X.to(device, non_blocking=True)
            logits = model(X)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    label_ids  = list(range(len(class_names)))

    acc        = accuracy_score(all_labels, all_preds)
    macro_f1   = f1_score(all_labels, all_preds, average="macro",
                          labels=label_ids, zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted",
                           labels=label_ids, zero_division=0)
    report     = classification_report(all_labels, all_preds, labels=label_ids,
                                       target_names=class_names, zero_division=0)

    print(f"\n  Test Accuracy : {acc:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Weighted F1   : {weighted_f1:.4f}")
    print(f"\n{report}")

    plot_confusion_matrix(all_labels, all_preds, class_names, output_dir)
    (output_dir / "classification_report.txt").write_text(report)
    metrics = {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}
    save_metrics(metrics, output_dir)
    return metrics
