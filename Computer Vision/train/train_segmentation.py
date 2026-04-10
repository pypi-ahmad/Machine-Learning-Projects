"""YOLO segmentation training pipeline.

Default base model: ``yolo26m-seg.pt`` (best accuracy for trainable
segmentation projects like aerial imagery, building footprint, road).

**Medical projects** (cell nuclei, lung, tumour, skin cancer) use MedSAM as
their primary segmentation engine and do NOT train through this pipeline.
If you want to compare YOLO-seg against MedSAM for a medical project, pass
``registry_project`` so the weights are registered and ``resolve()`` can
serve them alongside the MedSAM path.

Usage::

    from train.train_segmentation import train_segmentation

    # Standard YOLO-seg training
    results = train_segmentation(
        data_yaml="path/to/data.yaml",
        epochs=50,
    )

    # Override for a specific project
    results = train_segmentation(
        data_yaml="path/to/data.yaml",
        model="yolo26m-seg.pt",
        epochs=80,
        registry_project="building_footprint_segmentation",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("train.segmentation")


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


def train_segmentation(
    data_yaml: str,
    model: str = "yolo26m-seg.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    project: str = "runs/segment",
    name: str = "train",
    exist_ok: bool = True,
    registry_project: Optional[str] = None,
    registry_version: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train a YOLO segmentation model.

    Parameters
    ----------
    data_yaml : str
        Path to a YOLO-format ``data.yaml`` with segmentation masks.
    model : str
        Base segmentation model name or ``.pt`` path.
    epochs : int
        Number of training epochs.
    imgsz : int
        Input image size.
    batch : int
        Batch size.
    device : str, optional
        Device string.  Auto-detected if *None*.
    project : str
        Output directory.
    name : str
        Run name.
    exist_ok : bool
        Overwrite existing run.
    **kwargs
        Forwarded to ``model.train()``.

    Returns
    -------
    dict
        ``{"results": <Results>, "weights": <Path>}``
    """
    from ultralytics import YOLO

    data_yaml = str(Path(data_yaml).resolve())

    # Pre-flight: ensure dataset exists (attempt download if configured)
    if not Path(data_yaml).exists() and registry_project:
        _ensure_dataset(registry_project, data_yaml)

    log.info("Loading base model: %s", model)
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

    log.info("Starting segmentation training: %d epochs, imgsz=%d, batch=%d", epochs, imgsz, batch)
    results = yolo.train(**train_kwargs)

    best_weights = Path(project) / name / "weights" / "best.pt"
    log.info("Training complete. Best weights: %s", best_weights)

    # Auto-register in model registry
    if registry_project:
        _register_model(registry_project, registry_version, best_weights, results)

    return {"results": results, "weights": best_weights}


def _register_model(
    project: str, version: Optional[str], weights: Path, results: Any
) -> None:
    """Register trained weights in the model registry."""
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
        log.warning("Model registration failed (non-fatal): %s", exc)
