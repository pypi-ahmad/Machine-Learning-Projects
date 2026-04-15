"""High-level pipeline for Face Clustering Photo Organizer.

Orchestrates: scan images → detect faces → extract embeddings → cluster.

Usage::

    from parser import FaceClusterPipeline

    pipeline = FaceClusterPipeline(cfg)
    pipeline.load()
    clusters = pipeline.process(image_paths)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clusterer import Cluster, FaceClusterer, FaceRecord
from config import FaceClusterConfig
from embedder import FaceEmbedder
from face_detector import FaceDetector

log = logging.getLogger("face_cluster.parser")


@dataclass
class ClusterResult:
    """Complete pipeline result."""

    clusters: list[Cluster] = field(default_factory=list)
    total_images: int = 0
    total_faces: int = 0
    num_clusters: int = 0
    images_with_faces: int = 0
    images_without_faces: int = 0
    backend: str = ""


class FaceClusterPipeline:
    """Full face clustering pipeline.

    scan images → detect → embed → cluster
    """

    def __init__(self, cfg: FaceClusterConfig) -> None:
        self.cfg = cfg
        self.embedder = FaceEmbedder(cfg)
        self.detector = FaceDetector(cfg)
        self.clusterer = FaceClusterer(cfg)
        self._loaded = False

    def load(self) -> None:
        """Initialize all pipeline components."""
        insightface_app = self.embedder.load()
        det_backend = self.detector.load(insightface_app=insightface_app)
        self._loaded = True
        log.info(
            "Pipeline ready: det=%s, emb=%s",
            det_backend,
            "insightface" if self.embedder.ready else "none",
        )

    def process(self, image_paths: list[str | Path]) -> ClusterResult:
        """Process a collection of images.

        Parameters
        ----------
        image_paths : list
            Paths to images to scan.

        Returns
        -------
        ClusterResult
        """
        if not self._loaded:
            self.load()

        if not self.embedder.ready:
            return ClusterResult(backend="none")

        # Phase 1: detect + embed all faces
        all_records: list[FaceRecord] = []
        images_with = 0
        images_without = 0

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("Cannot read: %s", img_path)
                continue

            face_data = []
            detected_faces = self.detector.detect(img)
            for detected_face in detected_faces:
                embedding = self.embedder.extract_single(detected_face.crop)
                if embedding is None:
                    continue
                face_data.append({
                    "box": detected_face.box,
                    "confidence": detected_face.confidence,
                    "embedding": embedding,
                    "crop": detected_face.crop,
                })

            if not face_data and self.detector.backend == "insightface":
                face_data = self.embedder.extract_from_frame(img)

            if face_data:
                images_with += 1
                for fd in face_data:
                    all_records.append(FaceRecord(
                        source_path=str(img_path),
                        box=fd["box"],
                        confidence=fd["confidence"],
                        embedding=fd["embedding"],
                        crop=fd["crop"],
                    ))
            else:
                images_without += 1

        # Phase 2: cluster
        clusters = self.clusterer.cluster(all_records)

        return ClusterResult(
            clusters=clusters,
            total_images=len(image_paths),
            total_faces=len(all_records),
            num_clusters=len(clusters),
            images_with_faces=images_with,
            images_without_faces=images_without,
            backend=self.detector.backend or "none",
        )
