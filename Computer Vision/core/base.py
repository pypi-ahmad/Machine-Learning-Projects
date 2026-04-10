"""Base class for all modernized CV project wrappers.

Every ``modern.py`` subclasses :class:`CVProject` and implements at
minimum :meth:`load` and :meth:`predict`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class CVProject(ABC):
    """Abstract base for every modernized CV project."""

    # ── metadata (override in subclass) ────────────────────
    project_type: str = "generic"   # detection | segmentation | classification | pose | tracking | utility
    description: str = ""
    legacy_tech: str = ""           # what the original used (e.g. "Haar + Keras")
    modern_tech: str = ""           # what the v2 wrapper uses (e.g. "Ultralytics YOLO26")

    def __init__(self) -> None:
        self._loaded: bool = False
        self._load_time: float = 0.0

    # ── abstract interface ─────────────────────────────────
    @abstractmethod
    def load(self) -> None:
        """Load model weights and any required resources."""
        ...

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run inference on *input_data* (image path, numpy array, etc.)."""
        ...

    def visualize(self, input_data: Any, output: Any) -> Any:
        """Optional: return an annotated copy of *input_data*."""
        return output

    # ── built-in benchmarking ──────────────────────────────
    def benchmark(self, input_data: Any, *, n_runs: int = 10) -> dict:
        """Time :meth:`predict` over *n_runs* and return statistics."""
        if not self._loaded:
            t0 = time.perf_counter()
            self.load()
            self._load_time = time.perf_counter() - t0
            self._loaded = True

        latencies: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(input_data)
            latencies.append(time.perf_counter() - t0)

        arr = np.array(latencies)
        mean = float(arr.mean())
        return {
            "load_time_s": round(self._load_time, 4),
            "mean_latency_s": round(mean, 4),
            "std_latency_s": round(float(arr.std()), 4),
            "min_latency_s": round(float(arr.min()), 4),
            "max_latency_s": round(float(arr.max()), 4),
            "fps": round(1.0 / mean, 2) if mean > 0 else 0.0,
            "n_runs": n_runs,
        }
