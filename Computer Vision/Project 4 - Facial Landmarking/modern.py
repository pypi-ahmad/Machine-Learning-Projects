"""
Modern Facial Landmarking — MediaPipe Face Mesh v2
====================================================
Replaces legacy dlib 68-landmark detection with MediaPipe Face Landmarker.

Original: facial_landmarking.py (dlib HOG + shape_predictor_68)
Modern:   MediaPipe Face Mesh (468 dense face landmarks in real-time)

Note: YOLO-Pose COCO body keypoints only gives 5 face points (nose, eyes, ears).
      MediaPipe gives 468 face mesh landmarks — proper replacement for dlib 68.
      To stay in YOLO, custom-train YOLO26m-pose on a face-landmark dataset.

Usage:
    python -m core.runner --import-all facial_landmarks_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("facial_landmarks_v2")
class FacialLandmarksV2(CVProject):
    display_name = "Facial Landmarks (MediaPipe 468)"
    category = "pose"

    _face_mesh = None

    def load(self):
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_styles = mp.solutions.drawing_styles
            self._mp_face_mesh = mp.solutions.face_mesh
            print("  [facial_landmarks] MediaPipe Face Mesh loaded (468 landmarks)")
        except ImportError:
            print("  [facial_landmarks] MediaPipe not installed — install mediapipe")
            raise

    def predict(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._face_mesh.process(rgb)

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        vis = frame.copy()
        if output.multi_face_landmarks:
            for face_lms in output.multi_face_landmarks:
                self._mp_draw.draw_landmarks(
                    vis, face_lms,
                    self._mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_styles.get_default_face_mesh_tesselation_style(),
                )
                # Also draw contours for key features
                self._mp_draw.draw_landmarks(
                    vis, face_lms,
                    self._mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_styles.get_default_face_mesh_contours_style(),
                )
            n = len(output.multi_face_landmarks)
            cv2.putText(vis, f"Faces: {n} (468 landmarks)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis
