"""YOLO OBB (oriented bounding-box) training pipeline.

Default base model: ``yolo26m-obb.pt``.

Oriented bounding-box detection is used when objects appear at arbitrary
rotations in aerial / satellite imagery (ships, vehicles, aircraft, etc.).
Standard axis-aligned boxes waste area on rotated objects and reduce
IoU-based metric scores.

Usage::

    from train.train_obb import train_obb

    results = train_obb(
        data_yaml="path/to/data.yaml",
        epochs=100,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("train.obb")


def train_obb(
    data_yaml: str,
    model: str = "yolo26m-obb.pt",
    epochs: int = 100,
    imgsz: int = 1024,
    batch: int = 8,
    device: Optional[str] = None,
    project: str = "runs/obb",
    name: str = "train",
    exist_ok: bool = True,
    registry_project: Optional[str] = None,
    registry_version: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train a YOLO OBB model.

    Parameters
    ----------
    data_yaml : str
        Path to a YOLO-OBB-format ``data.yaml`` file.
    model : str
        Base OBB model name or path to ``.pt`` weights.
    epochs : int
        Number of training epochs.
    imgsz : int
        Input image size (pixels).  1024 is standard for aerial OBB.
    batch : int
        Batch size (lower default than detection due to larger images).
    device : str, optional
        Device string.  Auto-detected if *None*.
    project : str
        Output directory for training runs.
    name : str
        Run name inside *project*.
    exist_ok : bool
        Allow overwriting an existing run directory.
    **kwargs
        Extra keyword arguments forwarded to ``model.train()``.

    Returns
    -------
    dict
        ``{"results": <ultralytics Results>, "weights": <Path to best.pt>}``
    """
    from ultralytics import YOLO

    data_yaml = str(Path(data_yaml).resolve())

    log.info("Loading OBB base model: %s", model)
    yolo = YOLO(model)

    train_kwargs: Dict[str, Any] = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=exist_ok,
    )
    if device is not None:
        train_kwargs["device"] = device
    train_kwargs.update(kwargs)

    log.info("Starting OBB training: %d epochs, imgsz=%d, batch=%d", epochs, imgsz, batch)
    results = yolo.train(**train_kwargs)

    best_weights = Path(project) / name / "weights" / "best.pt"
    log.info("OBB training complete. Best weights: %s", best_weights)

    # Auto-register in model registry if available
    if registry_project:
        _register_model(registry_project, registry_version, best_weights, results)

    return {"results": results, "weights": best_weights}


def _register_model(
    project: str, version: Optional[str], weights: Path, results: Any
) -> None:
    """Register trained OBB weights in the model registry."""
    try:
        from models.registry import ModelRegistry

        reg = ModelRegistry()
        ver = version or f"v{len(reg.list_versions(project)) + 1}"

        metrics: Dict[str, Any] = {}
        if hasattr(results, "results_dict"):
            rd = results.results_dict
            for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
                if k in rd:
                    short = k.split("/")[-1].replace("(B)", "")
                    metrics[short] = round(float(rd[k]), 4)
        if not metrics:
            metrics["trained"] = True

        reg.register(project=project, version=ver, path=str(weights), metrics=metrics)
    except Exception as exc:
        log.warning("OBB model registration failed (non-fatal): %s", exc)
