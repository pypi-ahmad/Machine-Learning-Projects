"""Traffic Violation Analyzer — object tracker wrapper.

Wraps YOLO's built-in ``model.track(persist=True)`` and maintains a
per-track-ID position history so the rule engine can evaluate movement
direction and line crossings.

Usage::

    from tracker import TrackManager

    manager = TrackManager(max_history=30)
    detections = manager.update(yolo_results, names_map)
    trail = manager.get_trail(track_id)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from detector import Detection


class TrackManager:
    """Maintain per-ID position histories from YOLO tracking results."""

    def __init__(self, max_history: int = 60) -> None:
        self.max_history = max_history
        # track_id → list of (cx, cy) centers, newest last
        self._trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

    # ---- public API --------------------------------------------------------

    def update(
        self,
        results,
        conf_threshold: float = 0.30,
    ) -> list[Detection]:
        """Parse YOLO tracking results into :class:`Detection` list and
        record each tracked object's center.

        Parameters
        ----------
        results
            Raw YOLO results from ``model.track()``.
        conf_threshold
            Minimum confidence to keep.

        Returns
        -------
        list[Detection]
            Detections with ``track_id`` populated (when available).
        """
        detections: list[Detection] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue

            has_ids = boxes.id is not None
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                tid = int(boxes.id[i]) if has_ids else None

                det = Detection(
                    box=(x1, y1, x2, y2),
                    center=(cx, cy),
                    class_name=cls_name,
                    confidence=conf,
                    class_id=cls_id,
                    track_id=tid,
                )
                detections.append(det)

                # Record trail
                if tid is not None:
                    trail = self._trails[tid]
                    trail.append((cx, cy))
                    if len(trail) > self.max_history:
                        del trail[: len(trail) - self.max_history]

        return detections

    def get_trail(self, track_id: int) -> list[tuple[int, int]]:
        """Return the position history for *track_id* (may be empty)."""
        return list(self._trails.get(track_id, []))

    def get_all_trails(self) -> dict[int, list[tuple[int, int]]]:
        """Return a copy of all active trails."""
        return {k: list(v) for k, v in self._trails.items()}

    def clear(self) -> None:
        """Reset all tracked trails."""
        self._trails.clear()
