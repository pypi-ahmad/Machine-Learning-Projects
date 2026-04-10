"""Classification training pipeline — transfer learning with torchvision.

Supports ResNet-18/34/50, EfficientNet-B0, MobileNet-V2 out of the box.

**Baseline model strategy:**
    The classification baseline is **ResNet-18** (transfer learning from
    ImageNet).  This is the model that ``train_classification()`` trains by
    default and produces a custom ``best_model.pt`` registered in the
    ModelRegistry.

    ``yolo26m-cls.pt`` is NEVER trained here.  It exists only as a
    zero-config **inference fallback** — ``resolve(project, "cls")`` returns
    it when no custom weights are registered, giving every classification
    project a working pretrained model out of the box.  Once training runs
    and registers custom ResNet weights, ``resolve()`` returns those instead.

    To change the training baseline, pass ``model_name="efficientnet_b0"`` or
    any other key from ``_MODEL_REGISTRY``.

Usage::

    from train.train_classification import train_classification

    stats = train_classification(
        data_dir="data/emotion_recognition",
        num_classes=7,
        epochs=25,
    )
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

log = logging.getLogger("train.classification")


def _ensure_dataset(project_key: str, data_path: str) -> None:
    """Attempt auto-download of the dataset via the config-based downloader."""
    try:
        from utils.data_downloader import ensure_dataset
        log.info("Dataset missing at %s — attempting auto-download for '%s'", data_path, project_key)
        result = ensure_dataset(project_key)
        if result.get("ok"):
            log.info("Dataset downloaded successfully via %s", result.get("source", "?"))
        else:
            log.warning("Auto-download failed: %s (status=%s)", result.get("error", "?"), result.get("status", "?"))
    except ImportError:
        log.debug("data_downloader not available; skipping auto-download")

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: Dict[str, Any] = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "mobilenet_v2": models.mobilenet_v2,
}


def _build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a torchvision model and replace the final layer."""
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_MODEL_REGISTRY)}")

    weights = "DEFAULT" if pretrained else None
    model = _MODEL_REGISTRY[model_name](weights=weights)

    # Replace classifier head
    if model_name.startswith("resnet"):
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
    elif model_name == "efficientnet_b0":
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, num_classes)
    elif model_name == "mobilenet_v2":
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, num_classes)

    return model


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _build_transforms(imgsz: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, val_transform)."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(imgsz),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(imgsz + 32),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def _find_split_dir(data_dir: Path, candidates: List[str]) -> Optional[Path]:
    """Return the first existing subdirectory from *candidates*."""
    for name in candidates:
        p = data_dir / name
        if p.is_dir():
            return p
    return None


def _build_dataloaders(
    data_dir: str,
    imgsz: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader, List[str], int]:
    """Build train/val DataLoaders from an ImageFolder-compatible directory.

    Returns (train_loader, val_loader, class_names, num_classes).
    """
    data_path = Path(data_dir)
    train_tf, val_tf = _build_transforms(imgsz)

    train_dir = _find_split_dir(data_path, ["train", "training", "Train"])
    val_dir = _find_split_dir(data_path, ["val", "validation", "valid", "test", "Val", "Test"])

    if train_dir is not None:
        # Explicit split directories
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
        class_names = train_ds.classes
        if val_dir is not None:
            val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
        else:
            # Split training set
            n_train = int((1 - val_split) * len(train_ds))
            n_val = len(train_ds) - n_train
            train_ds, val_ds = random_split(train_ds, [n_train, n_val])
    else:
        # No split dirs — treat data_dir itself as ImageFolder
        full_ds = datasets.ImageFolder(str(data_path), transform=train_tf)
        class_names = full_ds.classes
        n_train = int((1 - val_split) * len(full_ds))
        n_val = len(full_ds) - n_train
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])

        # Give val subset proper val-time transforms
        val_copy = copy.copy(full_ds)
        val_copy.transform = val_tf
        val_ds.dataset = val_copy

    num_classes = len(class_names)
    log.info("Classes (%d): %s", num_classes, class_names[:10])
    log.info("Train samples: %d, Val samples: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, class_names, num_classes


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def train_classification(
    data_dir: str,
    num_classes: int = 0,
    model_name: str = "resnet18",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    imgsz: int = 224,
    device: Optional[str] = None,
    save_path: str = "best_model.pt",
    num_workers: int = 4,
    val_split: float = 0.2,
    pretrained: bool = True,
    registry_project: Optional[str] = None,
    registry_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a classification model using transfer learning.

    Parameters
    ----------
    data_dir : str
        Root directory in torchvision ImageFolder layout, or containing
        ``train/`` and ``val/`` sub-directories.
    num_classes : int
        Number of output classes. If ``0``, auto-detected from *data_dir*.
    model_name : str
        Torchvision model name (see ``_MODEL_REGISTRY``).
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Initial learning rate.
    imgsz : int
        Image size (square crop).
    device : str, optional
        ``"cpu"`` or ``"cuda"`` etc.  Auto-detected if *None*.
    save_path : str
        Where to save the best checkpoint.
    num_workers : int
        DataLoader workers.
    val_split : float
        Fraction held out for validation when no explicit split exists.
    pretrained : bool
        Use ImageNet pre-trained weights.

    Returns
    -------
    dict
        ``{"best_acc": float, "classes": list, "weights": Path, "history": list}``
    """
    # Device
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    log.info("Device: %s", dev)

    # Pre-flight: ensure dataset exists (attempt download if configured)
    if not Path(data_dir).exists() and registry_project:
        _ensure_dataset(registry_project, data_dir)

    # Data
    train_loader, val_loader, class_names, detected_classes = _build_dataloaders(
        data_dir, imgsz=imgsz, batch_size=batch_size,
        num_workers=num_workers, val_split=val_split,
    )
    if num_classes == 0:
        num_classes = detected_classes
        log.info("Auto-detected %d classes", num_classes)

    # Model
    model = _build_model(model_name, num_classes, pretrained=pretrained)
    model = model.to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train
    best_acc = 0.0
    history: List[Dict[str, float]] = []
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Training %s for %d epochs (classes=%d, lr=%g)", model_name, epochs, num_classes, lr)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_one_epoch(model, train_loader, criterion, optimizer, dev)
        val_loss, val_acc = _validate(model, val_loader, criterion, dev)
        scheduler.step()

        log.info(
            "Epoch %3d/%d — train_loss=%.4f  train_acc=%.3f  val_loss=%.4f  val_acc=%.3f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": class_names,
                "model_name": model_name,
                "num_classes": num_classes,
            }, save_path)
            log.info("  ✓ saved best model (val_acc=%.3f) → %s", val_acc, save_path)

    elapsed = time.time() - t0
    log.info("Done in %.1fs. Best val accuracy: %.3f", elapsed, best_acc)

    # Auto-register in model registry
    if registry_project:
        _register_cls_model(registry_project, registry_version, save_path, best_acc)

    return {
        "best_acc": best_acc,
        "classes": class_names,
        "weights": save_path,
        "history": history,
    }


def _register_cls_model(
    project: str,
    version: Optional[str],
    weights_path: Path,
    best_acc: float,
) -> None:
    """Register classification weights in the model registry."""
    try:
        from models.registry import ModelRegistry

        reg = ModelRegistry()
        ver = version or f"v{len(reg.list_versions(project)) + 1}"
        metrics = {"val_acc": round(best_acc, 4)}
        reg.register(project=project, version=ver, path=str(weights_path), metrics=metrics)
    except Exception as exc:
        log.warning("Model registration failed (non-fatal): %s", exc)
