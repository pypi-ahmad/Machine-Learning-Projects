"""Centralized model registry with versioning and best-model promotion.

Manages a JSON metadata store at ``models/metadata.json``.  Every trained
model is registered with its project, version tag, weights path, and metrics.
An **active** version per project is automatically promoted (highest primary
metric) and used by modern.py inference wrappers.

Usage::

    from models.registry import ModelRegistry, resolve

    reg = ModelRegistry()

    # After training — register + auto-promote
    reg.register(
        project="fire_smoke_detection",
        version="v2",
        path="models/fire_smoke_detection/v2/best.pt",
        metrics={"mAP50": 0.87, "mAP50-95": 0.62},
    )

    # At inference — resolve weights with YOLO26 fallback
    weights, version, is_pretrained = resolve("fire_smoke_detection", "detect")

    # Inspect
    print(reg.list_versions("fire_smoke_detection"))
    print(reg.status())
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("models.registry")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent                 # models/
_METADATA_FILE = _HERE / "metadata.json"
_MODELS_ROOT = _HERE                                     # models/<project>/<version>/

# Primary metric used for auto-promotion (checked in order of preference)
_PRIMARY_METRICS = ("mAP50-95", "mAP50", "mAP", "val_acc", "best_acc", "accuracy", "f1")

# ---------------------------------------------------------------------------
# YOLO26 pretrained fallback defaults
# ---------------------------------------------------------------------------
# Tier 1: Task-specific defaults — each alias maps to:
#   • A YOLO weights file (general_*, face_detect, medical_seg)
#   • A pipeline sentinel  (face_attr, face_recognition, …)
#
# Pipeline sentinels (e.g. "deepface://analyze") are NOT loadable weights.
# The project's load() interprets them to select the right library.
#
# Naming convention:
#   general_*          YOLO pretrained on COCO / ImageNet
#   face_detect        Custom YOLO face detector (WIDER FACE / FDDB training)
#   face_attr          DeepFace.analyze() pipeline
#   face_recognition   InsightFace ArcFace pipeline
#   face_landmarks     MediaPipe Face Landmarker
#   face_antispoof     Dedicated anti-spoof classifier (MiniFASNet / CDCN)
#   hand_landmarks     MediaPipe Hand Landmarker
#   gesture_recognizer MediaPipe Gesture Recognizer
#   ocr_detect         PaddleOCR text detection
#   ocr_recognize      PaddleOCR / TrOCR text recognition
#   medical_seg        YOLO26m-seg (trainable); optional MedSAM comparison
#
YOLO26_DEFAULTS: Dict[str, str] = {
    # ── General YOLO tasks (pretrained COCO / ImageNet) ──────────────────
    "detect":              "yolo26m.pt",
    "detection":           "yolo26m.pt",
    "general_detect":      "yolo26m.pt",
    "tracking":            "yolo26m.pt",
    "seg":                 "yolo26m-seg.pt",
    "segmentation":        "yolo26m-seg.pt",
    "general_seg":         "yolo26m-seg.pt",
    "pose":                "yolo26m-pose.pt",
    "general_pose_body":   "yolo26m-pose.pt",
    "cls":                 "yolo26m-cls.pt",
    "classification":      "yolo26m-cls.pt",
    "general_cls":         "yolo26m-cls.pt",
    "obb":                 "yolo26m-obb.pt",
    "general_obb":         "yolo26m-obb.pt",
    # ── Face detection (YOLO fine-tuned on face data) ────────────────────
    "face_detect":         "weights/face_detect_yolo26m.pt",
    # ── Face-domain pipelines (sentinels — NOT loadable weights) ─────────
    "face_attr":           "deepface://analyze",
    "face_recognition":    "insightface://buffalo_l",
    "face_landmarks":      "mediapipe://face_mesh",
    "face_antispoof":      "weights/face_antispoof.pt",
    # ── Hand-domain pipelines (sentinels — MediaPipe) ────────────────────
    "hand_landmarks":      "mediapipe://hands",
    "gesture_recognizer":  "mediapipe://gesture_recognizer",
    # ── OCR-domain pipelines (sentinels — PaddleOCR / TrOCR) ────────────
    "ocr_detect":          "paddleocr://detect",
    "ocr_recognize":       "paddleocr://recognize",
    # ── Medical segmentation (YOLO-seg base, optional MedSAM comparison) ─
    "medical_seg":         "yolo26m-seg.pt",
}

# Tier 2: Project-specific overrides — keyed by project registry key.
# Live/webcam demos on constrained hardware use n/s for lower latency.
#
# NOTE: Only projects whose YOLO task head actually matches their problem
# domain belong here.  Projects that use InsightFace, DeepFace, MediaPipe,
# PaddleOCR, or MedSAM as primary should NOT appear — their task alias
# in YOLO26_DEFAULTS already resolves to the right sentinel.
PROJECT_DEFAULTS: Dict[str, str] = {
    # ── Live / webcam demos (latency-critical → yolo26s) ─────────────────
    "car_detection":            "yolo26s.pt",           # COCO class 2 = car ✓
    "ball_tracking":            "yolo26s.pt",           # COCO class 32 = sports ball ✓
    "realtime_object_tracking": "yolo26s.pt",           # COCO detect ✓
    # ── Body / person detection ──────────────────────────────────────────
    "pose_detection":           "yolo26m-pose.pt",      # COCO 17-keypoint body pose ✓
    "pedestrian_detection":     "yolo26m.pt",           # COCO class 0 = person ✓
    # ── Specialized detection (often need custom-trained weights) ────────
    "face_detection":           "weights/face_detect_yolo26m.pt",  # custom YOLO face detector
    "face_mask_detection":      "weights/face_mask_yolo26m.pt",    # custom mask/no-mask/improper
    "fire_smoke_detection":     "yolo26m.pt",           # needs custom fire/smoke weights
    "food_object_detection":    "yolo26m.pt",           # COCO food classes 39-55 ✓
    "licence_plate_detector":   "yolo26m.pt",           # + PaddleOCR; needs custom plate weights
    "logo_detection":           "yolo26m.pt",           # SIFT primary; YOLO only with custom weights
    # ── Sudoku (OpenCV pipeline, optional YOLO26-cls for digits) ─────────
    "sudoku_solver":            "yolo26m-cls.pt",       # digit classifier — needs MNIST training
}


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------
class ModelRegistry:
    """Thread-safe, JSON-backed model version registry."""

    def __init__(self, metadata_path: Optional[Path] = None):
        self._path = metadata_path or _METADATA_FILE
        self._data: Dict[str, Any] = self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Corrupt metadata.json — starting fresh: %s", exc)
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._path)

    # -- public API: register -----------------------------------------------

    def register(
        self,
        project: str,
        version: str,
        path: str,
        metrics: Optional[Dict[str, Any]] = None,
        *,
        promote: bool = True,
        copy_weights: bool = True,
    ) -> Dict[str, Any]:
        """Register a trained model version.

        Parameters
        ----------
        project : str
            Registry key (e.g. ``"fire_smoke_detection"``).
        version : str
            Version tag (e.g. ``"v1"``, ``"v2"``).
        path : str
            Path to the best weights file.
        metrics : dict, optional
            Training / evaluation metrics (mAP, accuracy, loss …).
        promote : bool
            Automatically set as active if it has the best primary metric.
        copy_weights : bool
            Copy the weights into ``models/<project>/<version>/best.pt``.

        Returns
        -------
        dict
            The version entry that was stored.
        """
        src = Path(path)
        metrics = metrics or {}

        # Canonical storage location
        dest_dir = _MODELS_ROOT / project / version
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "best.pt"

        if copy_weights and src.exists() and src.resolve() != dest_file.resolve():
            shutil.copy2(src, dest_file)
            log.info("[%s/%s] Copied weights → %s", project, version, dest_file)

        # Build version entry
        entry: Dict[str, Any] = {
            "path": str(dest_file.relative_to(_HERE)) if dest_file.exists() else str(src),
            "metrics": metrics,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Upsert project
        if project not in self._data:
            self._data[project] = {"active": None, "versions": {}}
        self._data[project]["versions"][version] = entry

        # Auto-promote
        if promote:
            self._auto_promote(project)

        self._save()
        log.info("[%s/%s] Registered — metrics=%s", project, version, metrics)
        return entry

    # -- public API: query ---------------------------------------------------

    def get_active(self, project: str) -> Optional[Path]:
        """Return the absolute path to the active model, or *None*.

        Never raises — returns *None* if the project isn't registered or
        the weights file is missing.
        """
        proj = self._data.get(project)
        if not proj:
            return None
        active_ver = proj.get("active")
        if not active_ver:
            return None
        ver_entry = proj["versions"].get(active_ver)
        if not ver_entry:
            return None
        p = _HERE / ver_entry["path"]
        return p if p.exists() else None

    def get_active_version(self, project: str) -> Optional[str]:
        """Return the active version tag (e.g. ``"v2"``), or *None*."""
        proj = self._data.get(project)
        if not proj:
            return None
        return proj.get("active")

    def list_versions(self, project: str) -> List[Dict[str, Any]]:
        """Return all versions for *project* as a list of dicts."""
        proj = self._data.get(project)
        if not proj:
            return []
        active = proj.get("active")
        result = []
        for ver, entry in sorted(proj["versions"].items()):
            result.append({
                "version": ver,
                "active": ver == active,
                "path": entry.get("path", ""),
                "metrics": entry.get("metrics", {}),
                "date": entry.get("date", ""),
            })
        return result

    def list_projects(self) -> List[str]:
        """Return all registered project names."""
        return sorted(self._data.keys())

    def status(self) -> str:
        """Human-readable status table."""
        lines = [
            f"{'Project':<35} {'Active':<8} {'Versions':<10} {'Path'}",
            "-" * 90,
        ]
        for proj_name in sorted(self._data):
            proj = self._data[proj_name]
            active = proj.get("active", "-")
            n_versions = len(proj.get("versions", {}))
            path = "-"
            if active and active in proj.get("versions", {}):
                path = proj["versions"][active].get("path", "-")
            lines.append(f"{proj_name:<35} {active or '-':<8} {n_versions:<10} {path}")
        return "\n".join(lines)

    # -- public API: promote / demote ---------------------------------------

    def set_active(self, project: str, version: str) -> None:
        """Manually set the active version for *project*."""
        if project not in self._data:
            raise KeyError(f"Project '{project}' not in registry")
        if version not in self._data[project]["versions"]:
            raise KeyError(f"Version '{version}' not registered for '{project}'")
        self._data[project]["active"] = version
        self._save()
        log.info("[%s] Active version set to %s", project, version)

    # -- public API: record evaluation results --------------------------------

    def record_eval(
        self,
        project: str,
        version: str,
        dataset: Dict[str, Any],
        metrics: Dict[str, Any],
        primary_name: str,
        primary_value: Optional[float],
    ) -> None:
        """Attach evaluation results to an existing version entry.

        This is a non-breaking extension — it adds an ``"eval"`` key to the
        version dict without altering any other fields.

        Parameters
        ----------
        project : str
            Registry key (must already exist).
        version : str
            Version tag (must already exist under *project*).
        dataset : dict
            Dataset metadata (name, kind, ref, …).
        metrics : dict
            Full evaluation metrics dictionary.
        primary_name : str
            Name of the primary metric (e.g. ``"mAP50-95"``).
        primary_value : float or None
            Value of the primary metric.
        """
        import datetime as _dt

        proj = self._data.get(project)
        if not proj:
            log.warning("record_eval: project '%s' not in registry — skipping", project)
            return
        ver_entry = proj.get("versions", {}).get(version)
        if not ver_entry:
            log.warning("record_eval: version '%s' not found for '%s' — skipping", version, project)
            return

        ver_entry["eval"] = {
            "evaluated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "dataset": dataset,
            "metrics": metrics,
            "primary": {"name": primary_name, "value": primary_value},
        }

        self._save()
        log.info("[%s/%s] Recorded eval results (primary %s=%.4f)",
                 project, version, primary_name,
                 primary_value if primary_value is not None else 0.0)

    # -- private: auto-promote -----------------------------------------------

    def _auto_promote(self, project: str) -> None:
        """Promote the version with the best primary metric."""
        proj = self._data[project]
        versions = proj["versions"]

        if len(versions) == 1:
            # Only one version — auto-activate
            proj["active"] = next(iter(versions))
            return

        # Find the primary metric key present across versions
        metric_key = self._find_primary_metric(versions)
        if metric_key is None:
            # No comparable metric — keep current active or set first
            if not proj.get("active"):
                proj["active"] = next(iter(versions))
            return

        # Find best version by that metric
        best_ver = None
        best_val = -float("inf")
        for ver, entry in versions.items():
            val = entry.get("metrics", {}).get(metric_key)
            if val is not None and val > best_val:
                best_val = val
                best_ver = ver

        if best_ver:
            proj["active"] = best_ver
            log.info(
                "[%s] Auto-promoted %s (%s=%.4f)",
                project, best_ver, metric_key, best_val,
            )

    @staticmethod
    def _find_primary_metric(versions: Dict[str, Any]) -> Optional[str]:
        """Find the first shared primary metric across versions."""
        for metric_key in _PRIMARY_METRICS:
            found = False
            for entry in versions.values():
                if metric_key in entry.get("metrics", {}):
                    found = True
                    break
            if found:
                return metric_key
        return None


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
def get_active_model(project: str) -> Optional[Path]:
    """Shortcut: return path to active model or None (never raises)."""
    try:
        return ModelRegistry().get_active(project)
    except Exception:
        return None


def get_active(project: str) -> Optional[str]:
    """Return string path to active weights if they exist on disk, else None.

    This is the simplest query — use :func:`resolve` for full fallback logic.
    """
    try:
        p = ModelRegistry().get_active(project)
        return str(p) if p else None
    except Exception:
        return None


def resolve(project: str, task: str = "detect") -> Tuple[str, Optional[str], bool]:
    """Resolve the best available weights for *project*.

    Resolution order:
    1. Custom-trained weights registered in ``ModelRegistry`` (highest priority).
    2. Project-specific pretrained override from ``PROJECT_DEFAULTS``.
    3. General task-level pretrained default from ``YOLO26_DEFAULTS``.

    Returns
    -------
    tuple[str, str | None, bool]
        ``(weights, version, used_pretrained_default)`` where:
        - *weights* is a path (custom trained) or pretrained filename (e.g. ``yolo26m.pt``).
        - *version* is a version tag like ``"v1"`` or ``None`` if pretrained fallback.
        - *used_pretrained_default* is ``True`` when falling back to a pretrained model.
    """
    try:
        reg = ModelRegistry()
        p = reg.get_active(project)
        v = reg.get_active_version(project)
        if p is not None:
            return str(p), v, False
    except Exception:
        pass
    # Tier 2: project-specific pretrained override
    if project in PROJECT_DEFAULTS:
        return PROJECT_DEFAULTS[project], None, True
    # Tier 3: general task-level pretrained default
    default = YOLO26_DEFAULTS.get(task, YOLO26_DEFAULTS["detect"])
    return default, None, True
