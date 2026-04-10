"""Modern v2 pipeline — Face Landmark Detection.

Replaces: dlib HOG face detector + shape_predictor_68
Uses:     MediaPipe Face Landmarker (468 face mesh landmarks)

Pipeline: MediaPipe detects face + returns 468 dense landmarks in real-time.
          If you want to stay in YOLO, custom-train YOLO26m-pose on a
          face-landmark dataset — default COCO pose only gives 5 face points.

The original dlib implementation is preserved in faceLandmark.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("face_landmark_detection")
class FaceLandmarkModern(CVProject):
    project_type = "pose"
    description = "Face landmark detection (468 mesh landmarks via MediaPipe)"
    legacy_tech = "dlib HOG + shape_predictor_68_face_landmarks"
    modern_tech = "MediaPipe Face Landmarker (468 landmarks)"

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
            print("  [face_landmark] MediaPipe Face Mesh loaded (468 landmarks)")
        except ImportError:
            print("  [face_landmark] MediaPipe not installed — install mediapipe")
            raise

    def predict(self, input_data):
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        return results

    def visualize(self, input_data, output):
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        vis = frame.copy()
        if output.multi_face_landmarks:
            for face_lms in output.multi_face_landmarks:
                self._mp_draw.draw_landmarks(
                    vis, face_lms,
                    self._mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_styles.get_default_face_mesh_tesselation_style(),
                )
            n = len(output.multi_face_landmarks)
            cv2.putText(vis, f"Faces: {n} (468 landmarks each)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis
