"""
Modern Finger Counter — MediaPipe Hands v2
=============================================
Replaces legacy MediaPipe hand tracking with proper finger-counting logic.

Original: fingerCount.py + HandDetectionModule.py (MediaPipe Hands)
Modern:   MediaPipe Hand Landmarker (21 hand keypoints) + finger-state logic

Pipeline: detect hands → extract 21 landmarks → determine which fingers
          are extended based on landmark positions → count and display.

Note: YOLO-Pose COCO gives only wrist-level points (no fingers).
      MediaPipe gives 21 hand keypoints per hand — correct for finger counting.

Usage:
    python -m core.runner --import-all finger_counter_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


# MediaPipe hand landmark indices for fingertip and pip joints
_TIP_IDS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
_PIP_IDS = [3, 6, 10, 14, 18]   # corresponding pip/ip joints


def _count_fingers(hand_landmarks, handedness: str) -> int:
    """Count extended fingers from MediaPipe hand landmarks."""
    lm = hand_landmarks.landmark
    count = 0

    # Thumb: compare tip x vs ip x (accounts for left/right hand)
    if handedness == "Right":
        if lm[_TIP_IDS[0]].x < lm[_PIP_IDS[0]].x:
            count += 1
    else:
        if lm[_TIP_IDS[0]].x > lm[_PIP_IDS[0]].x:
            count += 1

    # Other 4 fingers: tip y < pip y means extended (image coords: up = smaller y)
    for i in range(1, 5):
        if lm[_TIP_IDS[i]].y < lm[_PIP_IDS[i]].y:
            count += 1

    return count


@register("finger_counter_v2")
class FingerCounterV2(CVProject):
    display_name = "Finger Counter (MediaPipe Hands)"
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
            print("  [finger_counter] MediaPipe Hands loaded (21 landmarks per hand)")
        except ImportError:
            print("  [finger_counter] MediaPipe not installed — install mediapipe")
            raise

    def predict(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        hands_data = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label  # "Left" or "Right"
                fingers = _count_fingers(hand_lms, label)
                hands_data.append({
                    "landmarks": hand_lms,
                    "handedness": label,
                    "fingers_up": fingers,
                })
        return {"hands": hands_data, "raw": results}

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        vis = frame.copy()
        total = 0
        for h in output.get("hands", []):
            self._mp_draw.draw_landmarks(
                vis, h["landmarks"],
                self._mp_hands.HAND_CONNECTIONS,
            )
            total += h["fingers_up"]

        # Display total finger count
        cv2.putText(vis, f"Fingers: {total}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        for i, h in enumerate(output.get("hands", [])):
            y = 90 + i * 30
            cv2.putText(vis, f"{h['handedness']}: {h['fingers_up']}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return vis
