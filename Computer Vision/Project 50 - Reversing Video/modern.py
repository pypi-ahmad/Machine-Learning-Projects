"""
Modern Video Reversing — CVProject wrapper v2
===============================================
Wraps video reverse playback in the unified CVProject framework.

Original: reversing video (openCV).ipynb
Modern:   Frame buffer reversal, unified interface

Usage:
    python -m core.runner --import-all video_reverse_v2 --source video.mp4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from collections import deque

from core.base import CVProject
from core.registry import register


@register("video_reverse_v2")
class VideoReverseV2(CVProject):
    display_name = "Video Reverse (v2)"
    category = "opencv_utility"

    BUFFER_SIZE = 60  # frames

    def load(self):
        self.frame_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.playback_idx = 0
        self.recording = True

    def predict(self, frame: np.ndarray):
        if self.recording:
            self.frame_buffer.append(frame.copy())
            if len(self.frame_buffer) >= self.BUFFER_SIZE:
                self.recording = False
                self.playback_idx = len(self.frame_buffer) - 1
            return {"mode": "recording", "count": len(self.frame_buffer)}
        else:
            if self.playback_idx >= 0:
                rev_frame = self.frame_buffer[self.playback_idx]
                self.playback_idx -= 1
                return {"mode": "playing", "frame": rev_frame,
                        "idx": self.playback_idx + 1}
            else:
                self.recording = True
                self.frame_buffer.clear()
                return {"mode": "recording", "count": 0}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        if output["mode"] == "recording":
            annotated = frame.copy()
            cv2.putText(annotated, f"Recording... {output['count']}/{self.BUFFER_SIZE}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(annotated, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        else:
            annotated = output.get("frame", frame).copy()
            cv2.putText(annotated, f"Reverse playback [{output.get('idx', 0)}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated
