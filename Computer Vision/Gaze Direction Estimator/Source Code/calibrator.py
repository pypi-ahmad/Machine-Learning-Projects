"""Optional calibration for Gaze Direction Estimator.

Captures baseline iris ratios while the user looks at five
positions (center, left, right, up, down) to compute personal
offsets.  These offsets are then applied during classification
to improve accuracy for a given user + camera setup.

Usage::

    calibrator = GazeCalibrator(cfg)
    # During calibration loop:
    calibrator.record(iris_position, "CENTER")
    # After all 5 positions:
    offsets = calibrator.compute_offsets()
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GazeConfig
from iris_locator import IrisPosition

log = logging.getLogger("gaze.calibrator")

CALIBRATION_POSITIONS = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]


@dataclass
class CalibrationOffsets:
    """Calibration offsets applied to gaze ratios."""

    h_offset: float = 0.0
    v_offset: float = 0.0
    calibrated: bool = False

    # Per-position averages (for diagnostics)
    position_means: dict[str, tuple[float, float]] = field(
        default_factory=dict,
    )


class GazeCalibrator:
    """Collects iris ratio samples and computes calibration offsets."""

    def __init__(self, cfg: GazeConfig) -> None:
        self.cfg = cfg
        self._samples: dict[str, list[tuple[float, float]]] = {
            pos: [] for pos in CALIBRATION_POSITIONS
        }

    def record(self, iris: IrisPosition, position: str) -> None:
        """Record one iris sample for the given gaze position.

        Parameters
        ----------
        iris : IrisPosition
            Current iris ratios.
        position : str
            One of CENTER, LEFT, RIGHT, UP, DOWN.
        """
        if position not in self._samples:
            return
        if iris.detected:
            self._samples[position].append((iris.h_ratio, iris.v_ratio))

    def has_enough(self, position: str) -> bool:
        """Check if enough frames have been collected for a position."""
        return len(self._samples.get(position, [])) >= self.cfg.calibration_frames

    def all_complete(self) -> bool:
        """Check if all positions have enough samples."""
        return all(self.has_enough(p) for p in CALIBRATION_POSITIONS)

    def compute_offsets(self) -> CalibrationOffsets:
        """Compute calibration offsets from collected samples.

        The offset is the difference between the measured center
        ratios and the ideal center (0.5, 0.5).

        Returns
        -------
        CalibrationOffsets
        """
        offsets = CalibrationOffsets()

        center_samples = self._samples.get("CENTER", [])
        if not center_samples:
            log.warning("No CENTER samples — calibration skipped")
            return offsets

        # Compute mean ratio at each position
        for pos, samples in self._samples.items():
            if samples:
                arr = np.array(samples)
                offsets.position_means[pos] = (
                    float(arr[:, 0].mean()),
                    float(arr[:, 1].mean()),
                )

        # Offset = measured_center - ideal_center
        center_h, center_v = offsets.position_means["CENTER"]
        offsets.h_offset = center_h - 0.5
        offsets.v_offset = center_v - 0.5
        offsets.calibrated = True

        log.info(
            "Calibration done: h_offset=%.3f, v_offset=%.3f",
            offsets.h_offset, offsets.v_offset,
        )
        return offsets

    def save(self, path: str | Path, offsets: CalibrationOffsets) -> None:
        """Save calibration offsets to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "h_offset": offsets.h_offset,
            "v_offset": offsets.v_offset,
            "calibrated": offsets.calibrated,
            "position_means": {
                k: list(v) for k, v in offsets.position_means.items()
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("Calibration saved → %s", path)

    @staticmethod
    def load(path: str | Path) -> CalibrationOffsets:
        """Load calibration offsets from a JSON file."""
        path = Path(path)
        if not path.exists():
            return CalibrationOffsets()

        data = json.loads(path.read_text(encoding="utf-8"))
        offsets = CalibrationOffsets(
            h_offset=data.get("h_offset", 0.0),
            v_offset=data.get("v_offset", 0.0),
            calibrated=data.get("calibrated", False),
            position_means={
                k: tuple(v)
                for k, v in data.get("position_means", {}).items()
            },
        )
        return offsets

    def reset(self) -> None:
        """Clear all collected samples."""
        for pos in self._samples:
            self._samples[pos].clear()
