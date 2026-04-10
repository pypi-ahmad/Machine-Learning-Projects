"""YOLO **custom-keypoint** pose-estimation training pipeline.

Use this instead of ``train_pose.py`` when your keypoint layout differs from
the standard COCO 17-body-keypoint skeleton.  Examples:

- **Animal pose** (e.g. AP-10K: 17 keypoints, different semantics)
- **Vehicle keypoints** (corner/wheel markers)
- **Custom object** keypoint annotations

For **face landmarks** and **hand landmarks**, you should use MediaPipe
(pretrained, no training needed) — not a YOLO-pose model.

Usage::

    from train.train_pose_custom import train_pose_custom

    results = train_pose_custom(
        data_yaml="path/to/data.yaml",
        model="yolo26m-pose.pt",  # fine-tune from COCO-17 body weights
        kpt_shape=[21, 3],        # e.g. 21 keypoints with (x, y, visibility)
        epochs=100,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("train.pose_custom")


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


def train_pose_custom(
    data_yaml: str,
    model: str = "yolo26m-pose.pt",
    kpt_shape: Optional[List[int]] = None,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    project: str = "runs/pose_custom",
    name: str = "train",
    exist_ok: bool = True,
    registry_project: Optional[str] = None,
    registry_version: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train a YOLO pose model with a custom keypoint layout.

    Parameters
    ----------
    data_yaml : str
        Path to a YOLO-format ``data.yaml`` with keypoint annotations.
        The ``data.yaml`` must define ``kpt_shape`` (e.g. ``[21, 3]``) to
        match your annotation format.
    model : str
        Base model to fine-tune.  Defaults to the COCO-17 body-pose
        checkpoint, which will be adapted to your custom keypoint count.
    kpt_shape : list[int], optional
        Explicit keypoint shape ``[num_keypoints, dims]``.  If provided,
        overrides the value in ``data.yaml``.  ``dims`` is typically 2
        ``(x, y)`` or 3 ``(x, y, visibility)``.
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

    if not Path(data_yaml).exists() and registry_project:
        _ensure_dataset(registry_project, data_yaml)

    log.info("Loading base model for custom-keypoint training: %s", model)
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
    if kpt_shape is not None:
        train_kwargs["kpt_shape"] = kpt_shape
        log.info("Custom kpt_shape: %s", kpt_shape)
    if device is not None:
        train_kwargs["device"] = device
    train_kwargs.update(kwargs)

    log.info("Starting custom-pose training: %d epochs, imgsz=%d, batch=%d", epochs, imgsz, batch)
    results = yolo.train(**train_kwargs)

    best_weights = Path(project) / name / "weights" / "best.pt"
    log.info("Training complete. Best weights: %s", best_weights)

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
