"""Modern v2 pipeline — Sign Language Recognition.

Replaces: MediaPipe Hands + custom Keras LSTM
Uses:     MediaPipe Holistic (hand + face + body landmarks) for feature extraction

Pipeline:
  Static signs:  MediaPipe Hand Landmarker / Gesture Recognizer
  Dynamic signs: hand + face + body landmarks → temporal classifier (LSTM/Transformer)

Note: Default YOLO-Pose gives body-only keypoints (no hand/face detail).
      Sign language needs hand landmarks (21 per hand) + face landmarks
      for non-manual signals + body pose for spatial reference.

The original MediaPipe + Keras implementation is preserved in app.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("sign_language_recognition")
class SignLanguageModern(CVProject):
    project_type = "pose"
    description = "Sign language recognition (hand + face + body landmarks)"
    legacy_tech = "MediaPipe Hands + Keras LSTM"
    modern_tech = "MediaPipe Holistic (hands + face + pose) + temporal classifier"

    _holistic = None

    def load(self):
        try:
            import mediapipe as mp
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_holistic = mp.solutions.holistic
            self._mp_styles = mp.solutions.drawing_styles
            print("  [sign_language] MediaPipe Holistic loaded (hands + face + pose)")
            print("  [sign_language] For temporal sign recognition, train an LSTM/Transformer on landmark sequences")
        except ImportError:
            print("  [sign_language] MediaPipe not installed — install mediapipe")
            raise

    def predict(self, input_data):
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(rgb)

        # Extract landmark arrays for downstream temporal model
        landmarks = {}
        if results.left_hand_landmarks:
            landmarks["left_hand"] = [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark]
        if results.right_hand_landmarks:
            landmarks["right_hand"] = [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark]
        if results.face_landmarks:
            landmarks["face"] = [(lm.x, lm.y, lm.z) for lm in results.face_landmarks.landmark]
        if results.pose_landmarks:
            landmarks["pose"] = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

        return {"holistic": results, "landmarks": landmarks}

    def visualize(self, input_data, output):
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        vis = frame.copy()
        results = output.get("holistic")
        if results is None:
            return vis

        # Draw pose skeleton
        if results.pose_landmarks:
            self._mp_draw.draw_landmarks(
                vis, results.pose_landmarks,
                self._mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
            )
        # Draw hand landmarks
        if results.left_hand_landmarks:
            self._mp_draw.draw_landmarks(
                vis, results.left_hand_landmarks,
                self._mp_holistic.HAND_CONNECTIONS,
            )
        if results.right_hand_landmarks:
            self._mp_draw.draw_landmarks(
                vis, results.right_hand_landmarks,
                self._mp_holistic.HAND_CONNECTIONS,
            )
        # Draw face mesh (light)
        if results.face_landmarks:
            self._mp_draw.draw_landmarks(
                vis, results.face_landmarks,
                self._mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._mp_styles.get_default_face_mesh_contours_style(),
            )

        # Status overlay
        lm = output.get("landmarks", {})
        parts = []
        if "left_hand" in lm:
            parts.append("L-Hand")
        if "right_hand" in lm:
            parts.append("R-Hand")
        if "face" in lm:
            parts.append("Face")
        if "pose" in lm:
            parts.append("Pose")
        label = " | ".join(parts) if parts else "No landmarks"
        cv2.putText(vis, f"Sign Language: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return vis
