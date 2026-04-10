"""
Modern Non-Photorealistic Rendering — CVProject wrapper v2
============================================================
Wraps OpenCV NPR effects in the unified CVProject framework.

Original: Non-Photorealistic Rendering (openCV).ipynb
Modern:   Same core logic (stylization, edgePreservingFilter), unified interface

Usage:
    python -m core.runner --import-all npr_rendering_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("npr_rendering_v2")
class NPRRenderingV2(CVProject):
    display_name = "Non-Photorealistic Rendering (v2)"
    category = "opencv_utility"

    def load(self):
        pass

    def predict(self, frame: np.ndarray):
        stylized = cv2.stylization(frame, sigma_s=60, sigma_r=0.45)
        edge_preserved = cv2.edgePreservingFilter(frame, flags=1,
                                                   sigma_s=60, sigma_r=0.4)
        detail_enhanced = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        return {"stylized": stylized, "edge_preserved": edge_preserved,
                "detail_enhanced": detail_enhanced}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        h, w = frame.shape[:2]
        sz = (w // 2, h // 2)
        top = np.hstack([cv2.resize(frame, sz), cv2.resize(output["stylized"], sz)])
        bottom = np.hstack([cv2.resize(output["edge_preserved"], sz),
                            cv2.resize(output["detail_enhanced"], sz)])
        grid = np.vstack([top, bottom])
        cv2.putText(grid, "Original", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Stylized", (sz[0] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Edge Preserved", (5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(grid, "Detail Enhanced", (sz[0] + 5, sz[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return grid
