"""
Modern Motion Blurring — CVProject wrapper v2
===============================================
Wraps directional motion blur effect in the unified CVProject framework.

Original: motion blurring effect.ipynb
Modern:   Same core logic (kernel convolution), unified interface

Usage:
    python -m core.runner --import-all motion_blur_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("motion_blur_v2")
class MotionBlurV2(CVProject):
    display_name = "Motion Blur (v2)"
    category = "opencv_utility"

    KERNEL_SIZE = 30

    def load(self):
        # Horizontal motion blur kernel
        self.kernel_h = np.zeros((self.KERNEL_SIZE, self.KERNEL_SIZE))
        self.kernel_h[self.KERNEL_SIZE // 2, :] = np.ones(self.KERNEL_SIZE)
        self.kernel_h /= self.KERNEL_SIZE
        # Vertical motion blur kernel
        self.kernel_v = np.zeros((self.KERNEL_SIZE, self.KERNEL_SIZE))
        self.kernel_v[:, self.KERNEL_SIZE // 2] = np.ones(self.KERNEL_SIZE)
        self.kernel_v /= self.KERNEL_SIZE

    def predict(self, frame: np.ndarray):
        h_blur = cv2.filter2D(frame, -1, self.kernel_h)
        v_blur = cv2.filter2D(frame, -1, self.kernel_v)
        return {"horizontal": h_blur, "vertical": v_blur}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        h, w = frame.shape[:2]
        sz = (w // 3, h)
        row = np.hstack([
            cv2.resize(frame, sz),
            cv2.resize(output["horizontal"], sz),
            cv2.resize(output["vertical"], sz)
        ])
        cv2.putText(row, "Original", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(row, "H-Motion", (sz[0] + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(row, "V-Motion", (sz[0] * 2 + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return row
