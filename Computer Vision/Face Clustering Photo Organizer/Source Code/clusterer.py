"""Face clustering module for Face Clustering Photo Organizer.

Groups face embeddings into identity clusters using agglomerative
clustering or DBSCAN on cosine distances.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FaceClusterConfig

log = logging.getLogger("face_cluster.clusterer")


@dataclass
class FaceRecord:
    """One detected face with its source image and embedding."""

    source_path: str
    box: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray
    crop: np.ndarray


@dataclass
class Cluster:
    """A group of faces belonging to the same identity."""

    cluster_id: int
    faces: list[FaceRecord] = field(default_factory=list)
    centroid: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self.faces)

    def compute_centroid(self) -> np.ndarray:
        if not self.faces:
            self.centroid = np.zeros(512, dtype=np.float32)
        else:
            c = np.mean([f.embedding for f in self.faces], axis=0)
            norm = np.linalg.norm(c)
            self.centroid = c / norm if norm > 0 else c
        return self.centroid


class FaceClusterer:
    """Cluster face embeddings by identity."""

    def __init__(self, cfg: FaceClusterConfig) -> None:
        self.cfg = cfg

    def cluster(self, records: list[FaceRecord]) -> list[Cluster]:
        """Cluster face records into identity groups.

        Parameters
        ----------
        records : list[FaceRecord]
            All detected faces with embeddings.

        Returns
        -------
        list[Cluster]
            Identity clusters, sorted largest-first.
        """
        if len(records) < 2:
            if records:
                c = Cluster(cluster_id=0, faces=records)
                c.compute_centroid()
                return [c]
            return []

        embeddings = np.array([r.embedding for r in records], dtype=np.float32)

        if self.cfg.algorithm == "dbscan":
            labels = self._dbscan(embeddings)
        else:
            labels = self._agglomerative(embeddings)

        # Group into clusters
        cluster_map: dict[int, list[FaceRecord]] = {}
        for label, record in zip(labels, records):
            if label == -1:
                continue  # noise
            cluster_map.setdefault(label, []).append(record)

        # Filter small clusters and collect noise into singletons
        clusters: list[Cluster] = []
        cid = 0
        for label in sorted(cluster_map.keys()):
            faces = cluster_map[label]
            if len(faces) < self.cfg.min_cluster_size:
                continue
            c = Cluster(cluster_id=cid, faces=faces)
            c.compute_centroid()
            clusters.append(c)
            cid += 1

        # Merge very close clusters
        clusters = self._merge_close_clusters(clusters)

        # Sort by size descending
        clusters.sort(key=lambda c: c.size, reverse=True)

        # Re-number
        for i, c in enumerate(clusters):
            c.cluster_id = i

        log.info(
            "Clustered %d faces into %d identity groups",
            len(records), len(clusters),
        )
        return clusters

    def _agglomerative(self, embeddings: np.ndarray) -> np.ndarray:
        """Agglomerative clustering with cosine distance."""
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=self.cfg.distance_threshold,
        )
        labels = model.fit_predict(embeddings)
        log.debug(
            "Agglomerative: %d clusters (threshold=%.2f)",
            len(set(labels)), self.cfg.distance_threshold,
        )
        return labels

    def _dbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """DBSCAN clustering with cosine distance."""
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_distances

        dist_matrix = cosine_distances(embeddings)
        model = DBSCAN(
            eps=self.cfg.dbscan_eps,
            min_samples=self.cfg.dbscan_min_samples,
            metric="precomputed",
        )
        labels = model.fit_predict(dist_matrix)
        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        log.debug(
            "DBSCAN: %d clusters, %d noise (eps=%.2f)",
            n_clusters, n_noise, self.cfg.dbscan_eps,
        )
        return labels

    def _merge_close_clusters(
        self, clusters: list[Cluster],
    ) -> list[Cluster]:
        """Merge clusters whose centroids are very close."""
        if len(clusters) < 2:
            return clusters

        merged = list(clusters)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(merged):
                j = i + 1
                while j < len(merged):
                    ci = merged[i].centroid
                    cj = merged[j].centroid
                    if ci is None or cj is None:
                        j += 1
                        continue
                    cos_dist = 1.0 - float(
                        np.dot(ci, cj)
                        / (np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-8)
                    )
                    if cos_dist < self.cfg.merge_threshold:
                        # Merge j into i
                        merged[i].faces.extend(merged[j].faces)
                        merged[i].compute_centroid()
                        merged.pop(j)
                        changed = True
                    else:
                        j += 1
                i += 1

        return merged
