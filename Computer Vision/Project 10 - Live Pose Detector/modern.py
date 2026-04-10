"""
Modern Pose Detector — YOLO26m-Pose v2
========================================
Replaces legacy MediaPipe Pose + FaceMesh with YOLO26m-Pose.

Original: poseDetector-I.py (MediaPipe holistic: pose + face + hands)
Modern:   YOLO26m-Pose (17 COCO body keypoints)

This is the one pose project that correctly matches what YOLO-Pose does:
full-body human pose estimation with 17 COCO keypoints.

Usage:
    python -m core.runner --import-all pose_detector_v2 --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo


# COCO keypoint names for reference
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@register("pose_detector_v2")
class PoseDetectorV2(CVProject):
    display_name = "Pose Detector (YOLO26m-Pose)"
    category = "pose"

    CONF_THRESHOLD = 0.5

    def load(self):
        _key, _task = "pose_detection", "pose"
        from models.registry import resolve
        weights, version, is_default = resolve(_key, _task)
        print(f"  [{_key}] version={version}  weights={weights}  pretrained_fallback={is_default}")
        self.model = load_yolo(weights)

    def predict(self, frame: np.ndarray):
        results = self.model(frame, verbose=False, conf=self.CONF_THRESHOLD)
        return results

    def visualize(self, frame: np.ndarray, output) -> np.ndarray:
        annotated = output[0].plot()
        keypoints = output[0].keypoints
        if keypoints is not None and len(keypoints) > 0:
            n_persons = len(keypoints.data)
            cv2.putText(
                annotated, f"Persons: {n_persons}",
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
        return annotated
