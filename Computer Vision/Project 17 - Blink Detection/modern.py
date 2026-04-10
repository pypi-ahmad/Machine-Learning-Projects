"""
Modern Blink Detection — MediaPipe Face Mesh + EAR v2
=======================================================
Replaces legacy dlib 68-landmark + EAR blink detection.

Original: blink_detector.py (dlib HOG + shape_predictor_68, EAR formula)
Modern:   MediaPipe Face Landmarker + Eye Aspect Ratio (EAR) thresholds

Pipeline: detect face mesh (468 landmarks) → extract 6 eye landmarks per eye
          → compute EAR → detect blink when EAR drops below threshold.

Note: YOLO-Pose COCO gives no eye landmarks — can't compute EAR.
      MediaPipe Face Mesh gives full iris/eyelid landmarks.

Usage:
    python -m core.runner --import-all blink_detection_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


# MediaPipe Face Mesh landmark indices for left/right eye contour
# These correspond to the 6-point EAR formulation
_LEFT_EYE = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 2


def _eye_aspect_ratio(landmarks, eye_indices, w, h):
    """Compute Eye Aspect Ratio from landmark coordinates."""
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))

    # Vertical distances
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    if h_dist < 1e-6:
        return 0.3
    return (v1 + v2) / (2.0 * h_dist)


@register("blink_detection_v2")
class BlinkDetectionV2(CVProject):
    display_name = "Blink Detection (MediaPipe + EAR)"
    category = "pose"

    _face_mesh = None
    _blink_count = 0
    _frame_counter = 0

    def load(self):
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_face_mesh = mp.solutions.face_mesh
            self._blink_count = 0
            self._frame_counter = 0
            print("  [blink_detection] MediaPipe Face Mesh + EAR blink detection loaded")
        except ImportError:
            print("  [blink_detection] MediaPipe not installed — install mediapipe")
            raise

    def predict(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        ear = None
        blinked = False
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            left_ear = _eye_aspect_ratio(lms, _LEFT_EYE, w, h)
            right_ear = _eye_aspect_ratio(lms, _RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                self._frame_counter += 1
            else:
                if self._frame_counter >= CONSEC_FRAMES:
                    self._blink_count += 1
                    blinked = True
                self._frame_counter = 0

        return {
            "face_results": results,
            "ear": ear,
            "blinked": blinked,
            "total_blinks": self._blink_count,
        }

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        vis = frame.copy()
        face_results = output.get("face_results")
        ear = output.get("ear")
        total = output.get("total_blinks", 0)

        if face_results and face_results.multi_face_landmarks:
            for face_lms in face_results.multi_face_landmarks:
                self._mp_draw.draw_landmarks(
                    vis, face_lms,
                    self._mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                )

        color = (0, 255, 0) if ear and ear >= EAR_THRESHOLD else (0, 0, 255)
        ear_str = f"{ear:.2f}" if ear is not None else "N/A"
        cv2.putText(vis, f"EAR: {ear_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis, f"Blinks: {total}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if output.get("blinked"):
            cv2.putText(vis, "BLINK!", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return vis
