"""Enrollment manager for Face Verification Attendance System.

Handles registering known identities: stores normalized embeddings
as a gallery on disk (JSON), supports multi-image enrollment with
mean embedding, and loads/saves gallery persistence.

Usage::

    from enrollment import EnrollmentManager

    mgr = EnrollmentManager(cfg, embedder)
    mgr.enroll("Alice", [img1, img2])
    mgr.save()
    mgr.load()
    gallery = mgr.gallery  # {name: np.ndarray}
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FaceAttendanceConfig

log = logging.getLogger("face_attendance.enrollment")


class EnrollmentManager:
    """Manage a gallery of enrolled face identities."""

    def __init__(self, cfg: FaceAttendanceConfig, embedder=None) -> None:
        self.cfg = cfg
        self._embedder = embedder
        self._gallery: dict[str, np.ndarray] = {}       # name → mean embedding
        self._raw: dict[str, list[np.ndarray]] = {}      # name → all embeddings

    @property
    def gallery(self) -> dict[str, np.ndarray]:
        """Read-only access to the gallery: ``{name: embedding}``."""
        return dict(self._gallery)

    @property
    def identities(self) -> list[str]:
        return sorted(self._gallery.keys())

    @property
    def size(self) -> int:
        return len(self._gallery)

    # ── enrollment ─────────────────────────────────────────

    def enroll(
        self,
        name: str,
        images: list[np.ndarray | str],
    ) -> bool:
        """Enroll an identity from one or more face images.

        Parameters
        ----------
        name : str
            Identity label (e.g., ``"Alice"``).
        images : list
            Face images (BGR arrays or file paths).

        Returns
        -------
        bool
            True if at least one face was successfully enrolled.
        """
        if self._embedder is None or not self._embedder.ready:
            log.error("Embedder not loaded — cannot enroll")
            return False

        embeddings: list[np.ndarray] = []
        for img in images[: self.cfg.max_enrollment_images]:
            frame = img if isinstance(img, np.ndarray) else cv2.imread(str(img))
            if frame is None:
                log.warning("Cannot read image for '%s'", name)
                continue
            emb = self._embedder.extract_single(frame)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) < self.cfg.min_enrollment_images:
            log.warning(
                "Not enough faces for '%s' (got %d, need %d)",
                name, len(embeddings), self.cfg.min_enrollment_images,
            )
            return False

        # Store raw and compute mean
        self._raw[name] = embeddings
        if self.cfg.use_mean_embedding and len(embeddings) > 1:
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)  # re-normalize
            self._gallery[name] = mean_emb
        else:
            self._gallery[name] = embeddings[0]

        log.info(
            "Enrolled '%s' with %d image(s) (dim=%d)",
            name, len(embeddings), self._gallery[name].shape[0],
        )
        return True

    def enroll_single(self, name: str, image) -> bool:
        """Convenience: enroll from a single image."""
        img = image if isinstance(image, np.ndarray) else str(image)
        return self.enroll(name, [img])

    def remove(self, name: str) -> bool:
        """Remove an identity from the gallery."""
        if name in self._gallery:
            del self._gallery[name]
            self._raw.pop(name, None)
            log.info("Removed '%s' from gallery", name)
            return True
        return False

    # ── persistence ────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> Path:
        """Save gallery to JSON.

        Parameters
        ----------
        path : str or Path, optional
            Output path. Defaults to ``<gallery_dir>/gallery.json``.

        Returns
        -------
        Path
            Path to the saved gallery file.
        """
        out = Path(path) if path else Path(self.cfg.gallery_dir) / "gallery.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        data = {
            name: emb.tolist()
            for name, emb in self._gallery.items()
        }
        out.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )
        log.info("Gallery saved (%d identities) → %s", len(data), out)
        return out

    def load(self, path: str | Path | None = None) -> bool:
        """Load gallery from JSON.

        Parameters
        ----------
        path : str or Path, optional
            Input path. Defaults to ``<gallery_dir>/gallery.json``.

        Returns
        -------
        bool
            True if gallery loaded successfully.
        """
        src = Path(path) if path else Path(self.cfg.gallery_dir) / "gallery.json"
        if not src.exists():
            log.warning("Gallery file not found: %s", src)
            return False

        data = json.loads(src.read_text(encoding="utf-8"))
        self._gallery = {
            name: np.array(emb, dtype=np.float32)
            for name, emb in data.items()
        }
        log.info("Gallery loaded (%d identities) ← %s", len(self._gallery), src)
        return True

    def set_gallery(self, gallery: dict[str, np.ndarray]) -> None:
        """Directly set the gallery (e.g. from external source)."""
        self._gallery = {
            k: v.astype(np.float32) for k, v in gallery.items()
        }
