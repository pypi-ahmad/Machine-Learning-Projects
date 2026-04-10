"""
Modern Volume Controller — MediaPipe Hands v2
================================================
Replaces legacy MediaPipe hand tracking with proper gesture-distance logic.

Original: VolumeControl.py + HandTrackingModule.py (MediaPipe Hands)
Modern:   MediaPipe Hand Landmarker + thumb-index distance → volume mapping

Pipeline: detect hand → extract 21 landmarks → compute thumb-tip to index-tip
          distance → map to volume range → display visual feedback.

Note: YOLO-Pose COCO gives only wrist-level points (no fingers).
      MediaPipe gives 21 hand keypoints — correct for gesture control.

Usage:
    python -m core.runner --import-all volume_controller_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register

# Hand landmark indices
_THUMB_TIP = 4
_INDEX_TIP = 8

# Distance range mapping (pixels) → volume percentage
_MIN_DIST = 20
_MAX_DIST = 200


@register("volume_controller_v2")
class VolumeControllerV2(CVProject):
    display_name = "Volume Controller (MediaPipe Hands)"
    category = "pose"

    _hands = None

    def load(self):
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_hands = mp.solutions.hands
            print("  [volume_controller] MediaPipe Hands loaded — thumb-index distance control")
        except ImportError:
            print("  [volume_controller] MediaPipe not installed — install mediapipe")
            raise

    def predict(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        vol_pct = None
        thumb_pos = None
        index_pos = None

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0].landmark
            thumb_pos = (int(lms[_THUMB_TIP].x * w), int(lms[_THUMB_TIP].y * h))
            index_pos = (int(lms[_INDEX_TIP].x * w), int(lms[_INDEX_TIP].y * h))
            dist = np.hypot(thumb_pos[0] - index_pos[0], thumb_pos[1] - index_pos[1])
            vol_pct = np.clip((dist - _MIN_DIST) / (_MAX_DIST - _MIN_DIST) * 100, 0, 100)

        return {
            "raw": results,
            "volume_pct": vol_pct,
            "thumb": thumb_pos,
            "index": index_pos,
        }

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        vis = frame.copy()
        raw = output.get("raw")

        if raw and raw.multi_hand_landmarks:
            for hand_lms in raw.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    vis, hand_lms,
                    self._mp_hands.HAND_CONNECTIONS,
                )

        thumb = output.get("thumb")
        index = output.get("index")
        vol = output.get("volume_pct")

        if thumb and index:
            cv2.circle(vis, thumb, 10, (255, 0, 0), -1)
            cv2.circle(vis, index, 10, (255, 0, 0), -1)
            cv2.line(vis, thumb, index, (0, 255, 0), 3)

        if vol is not None:
            # Volume bar
            bar_x, bar_y, bar_w, bar_h = 50, vis.shape[0] - 200, 30, 150
            fill_h = int(bar_h * vol / 100)
            cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
            cv2.rectangle(vis, (bar_x, bar_y + bar_h - fill_h),
                          (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), -1)
            cv2.putText(vis, f"Vol: {int(vol)}%", (bar_x - 10, bar_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "Show hand to control volume", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis
