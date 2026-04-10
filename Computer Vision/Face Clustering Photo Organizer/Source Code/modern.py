"""Modern registry entry for Face Clustering Photo Organizer.

Registers the project with the repo's ``core/registry.py`` decorator
so it can be discovered and launched via the unified CLI.

Pipeline: detect faces → InsightFace embeddings → cluster → organize.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("face_clustering_photo_organizer")
class FaceClusteringModern(CVProject):
    """Face clustering photo organizer — unsupervised identity grouping."""

    project_type = "detection"
    description = (
        "Face detection + InsightFace embeddings for unsupervised "
        "identity clustering and photo organization"
    )
    legacy_tech = "N/A (new project)"
    modern_tech = "InsightFace ArcFace embeddings + agglomerative/DBSCAN clustering"

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        self._validator = None
        self.cfg = None

    def load(self) -> None:
        from config import FaceClusterConfig
        from parser import FaceClusterPipeline
        from validator import ClusterValidator

        self.cfg = FaceClusterConfig()
        self._pipeline = FaceClusterPipeline(self.cfg)
        self._pipeline.load()
        self._validator = ClusterValidator(self.cfg)
        self._loaded = True

    def predict(self, input_data, **kwargs):
        if not self._loaded:
            self.load()

        # Accept a directory, list of paths, or single image
        if isinstance(input_data, (list, tuple)):
            paths = [str(p) for p in input_data]
        elif isinstance(input_data, np.ndarray):
            # Single frame — write to temp, process
            import tempfile
            tmp = Path(tempfile.mktemp(suffix=".jpg"))
            cv2.imwrite(str(tmp), input_data)
            paths = [str(tmp)]
        else:
            p = Path(str(input_data))
            if p.is_dir():
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                paths = sorted(
                    str(f) for f in p.rglob("*")
                    if f.suffix.lower() in exts
                )
            else:
                paths = [str(p)]

        result = self._pipeline.process(paths)
        report = self._validator.validate(result)

        return {
            "result": result,
            "report": report,
            "num_clusters": result.num_clusters,
            "total_faces": result.total_faces,
            "total_images": result.total_images,
        }

    def visualize(self, input_data, output, **kwargs):
        from collage import build_all_collages

        result = output["result"]
        collages = build_all_collages(result, self.cfg)
        if not collages:
            return np.zeros((100, 300, 3), dtype=np.uint8)
        # Stack all collages vertically
        imgs = [img for _, img in collages]
        max_w = max(img.shape[1] for img in imgs)
        padded = []
        for img in imgs:
            if img.shape[1] < max_w:
                pad = np.full(
                    (img.shape[0], max_w - img.shape[1], 3), 40,
                    dtype=np.uint8,
                )
                img = np.hstack([img, pad])
            padded.append(img)
        return np.vstack(padded)

    def setup(self, **kwargs) -> None:
        from config import FaceClusterConfig, load_config
        from parser import FaceClusterPipeline
        from validator import ClusterValidator

        config_path = kwargs.get("config")
        self.cfg = load_config(config_path) if config_path else FaceClusterConfig()
        if kwargs.get("algorithm"):
            self.cfg.algorithm = kwargs["algorithm"]
        if kwargs.get("threshold"):
            self.cfg.distance_threshold = kwargs["threshold"]
        self._pipeline = FaceClusterPipeline(self.cfg)
        self._pipeline.load()
        self._validator = ClusterValidator(self.cfg)
        self._loaded = True

    def train(self, **kwargs) -> None:
        from train import main as train_main
        train_main()

    def evaluate(self, **kwargs) -> None:
        from train import main as eval_main
        eval_main()
