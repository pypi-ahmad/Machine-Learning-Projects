"""Video Event Search — object tracker wrapper.

Wraps YOLO's ``model.track(persist=True)`` and maintains per-track-ID
position histories so the event generator can evaluate zone presence,
line crossings, dwell time, and appearance/disappearance.

Usage::

    from tracker import TrackManager

    manager = TrackManager(max_history=60)
    detections = manager.update(yolo_results, conf_threshold=0.3)
    trail = manager.get_trail(track_id)
"""

from __future__ import annotations

from collections import defaultdict

from detector import Detection


class TrackManager:
    """Per-ID position histories from YOLO tracking results."""

    def __init__(self, max_history: int = 120) -> None:
        self.max_history = max_history
        self._trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._active_ids: set[int] = set()
        self._prev_active_ids: set[int] = set()

    def update(
        self,
        results,
        conf_threshold: float = 0.30,
    ) -> list[Detection]:
        """Parse YOLO tracking results into :class:`Detection` list."""
        detections: list[Detection] = []
        current_ids: set[int] = set()

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

                if tid is not None:
                    current_ids.add(tid)
                    trail = self._trails[tid]
                    trail.append((cx, cy))
                    if len(trail) > self.max_history:
                        del trail[: len(trail) - self.max_history]

        self._prev_active_ids = self._active_ids
        self._active_ids = current_ids
        return detections

    @property
    def newly_appeared(self) -> set[int]:
        """Track IDs that appeared this frame but weren't active previously."""
        return self._active_ids - self._prev_active_ids

    @property
    def newly_disappeared(self) -> set[int]:
        """Track IDs active previously but gone this frame."""
        return self._prev_active_ids - self._active_ids

    def get_trail(self, track_id: int) -> list[tuple[int, int]]:
        return list(self._trails.get(track_id, []))

    def get_all_trails(self) -> dict[int, list[tuple[int, int]]]:
        return {k: list(v) for k, v in self._trails.items()}

    def is_active(self, track_id: int) -> bool:
        return track_id in self._active_ids

    def clear(self) -> None:
        self._trails.clear()
        self._active_ids.clear()
        self._prev_active_ids.clear()
