"""YOLO **body** pose-estimation training pipeline.

Default base model: ``yolo26m-pose.pt`` (COCO 17-keypoint body skeleton).

This pipeline is ONLY for body-pose projects (e.g. Project 10 — Pose
Detector).  **Do NOT use this for:**

- **Face landmarks** → Use MediaPipe Face Mesh (468 landmarks).
  No training needed; pretrained model is sufficient.
- **Hand landmarks** → Use MediaPipe Hands (21 keypoints).
  No training needed.
- **Custom keypoint** layouts (animal pose, vehicle keypoints, etc.)
  → Use ``train_pose_custom.py`` which accepts an explicit
  ``kpt_shape`` parameter instead of defaulting to COCO-17.

Usage::

    from train.train_pose import train_pose_body

    results = train_pose_body(
        data_yaml="path/to/data.yaml",
        epochs=50,
    )

Legacy alias ``train_pose`` is kept for backward compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("train.pose")


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


def train_pose_body(
    data_yaml: str,
    model: str = "yolo26m-pose.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    project: str = "runs/pose",
    name: str = "train",
    exist_ok: bool = True,
    registry_project: Optional[str] = None,
    registry_version: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train a YOLO body pose-estimation model (COCO 17-keypoint layout).

    Parameters
    ----------
    data_yaml : str
        Path to a YOLO-format ``data.yaml`` with keypoint annotations.
    model : str
        Base pose model name or ``.pt`` path.  Defaults to COCO-17 body pose.
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

    log.info("Loading base body-pose model: %s", model)
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

    log.info("Starting body-pose training: %d epochs, imgsz=%d, batch=%d", epochs, imgsz, batch)
    results = yolo.train(**train_kwargs)

    best_weights = Path(project) / name / "weights" / "best.pt"
    log.info("Training complete. Best weights: %s", best_weights)

    # Auto-register in model registry
    if registry_project:
        _register_model(registry_project, registry_version, best_weights, results)

    return {"results": results, "weights": best_weights}


# Backward-compatible alias
train_pose = train_pose_body


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
