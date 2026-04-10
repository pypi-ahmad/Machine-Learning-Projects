"""
Modern Hand Tracking Module — MediaPipe Hands v2
===================================================
Replaces legacy MediaPipe hand tracking wrapper with direct MediaPipe Hands.

Original: VolumeControl.py + HandTrackingModule.py (MediaPipe Hands)
Modern:   MediaPipe Hand Landmarker (21 keypoints per hand)

Note: YOLO-Pose COCO gives only body-level keypoints (wrists, no fingers).
      MediaPipe gives 21 hand keypoints — correct for hand tracking.
      Alternative: custom-train YOLO26m-pose on hand keypoint dataset.

Usage:
    python -m core.runner --import-all hand_tracking_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("hand_tracking_v2")
class HandTrackingV2(CVProject):
    display_name = "Hand Tracking (MediaPipe 21-landmark)"
    category = "pose"

    _hands = None

    def load(self):
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_hands = mp.solutions.hands
            print("  [hand_tracking] MediaPipe Hands loaded (21 landmarks per hand)")
        except ImportError:
            print("  [hand_tracking] MediaPipe not installed — install mediapipe")
            raise

    def predict(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        return results

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        vis = frame.copy()
        if output.multi_hand_landmarks:
            for hand_lms in output.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    vis, hand_lms,
                    self._mp_hands.HAND_CONNECTIONS,
                )
            n = len(output.multi_hand_landmarks)
            cv2.putText(vis, f"Hands: {n}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis
