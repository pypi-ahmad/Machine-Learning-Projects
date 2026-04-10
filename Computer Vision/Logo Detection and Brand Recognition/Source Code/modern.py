"""Modern v2 pipeline — Logo Detection and Brand Recognition.

Replaces: Notebook-only MobileNetV2 classifier
Uses:     YOLO26m detect + optional SIFT template matching for known logos

Pipeline:
  - With custom weights:  YOLO detect → logo-level bounding boxes
  - Without custom weights: SIFT/ORB matching against logo templates
    (set TEMPLATE_DIR to a folder of logo images)

Note: COCO has no logo class.  With pretrained COCO weights, YOLO cannot
detect logos.  Train on FlickrLogos-32, LogoDet-3K, or OpenLogo for
real logo detection.

The original notebook implementation is preserved in the .ipynb file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register
from utils.yolo import load_yolo
from models.registry import resolve


@register("logo_detection")
class LogoDetectionModern(CVProject):
    project_type = "detection"
    description = "Logo detection — YOLO custom weights or SIFT template matching"
    legacy_tech = "MobileNetV2 (notebook)"
    modern_tech = "YOLO26m detect (needs logo-trained weights) or SIFT matching"

    CONF_THRESHOLD = 0.3
    TEMPLATE_DIR = None  # Set to Path("path/to/logo_templates/") for SIFT mode
    _sift = None
    _templates = None
    _use_yolo = False

    def load(self):
        weights, ver, fallback = resolve("logo_detection", "detect")
        self._use_yolo = not fallback

        if self._use_yolo:
            self.model = load_yolo(weights)
            print(f"  [logo] Custom YOLO weights loaded: {weights}")
            return

        # No logo-trained weights — set up SIFT matching
        self._sift = cv2.SIFT_create()
        self._bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self._templates = []

        # Load logo templates from TEMPLATE_DIR (if set)
        if self.TEMPLATE_DIR and Path(self.TEMPLATE_DIR).is_dir():
            for f in Path(self.TEMPLATE_DIR).glob("*"):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        kp, des = self._sift.detectAndCompute(img, None)
                        if des is not None:
                            self._templates.append({"name": f.stem, "kp": kp, "des": des})
            print(f"  [logo] SIFT mode: {len(self._templates)} logo templates loaded")
        else:
            print("  [logo] No custom YOLO weights and no TEMPLATE_DIR set")
            print("  [logo] Train YOLO on logo dataset or set TEMPLATE_DIR for SIFT matching")
            # Load YOLO as generic fallback
            self.model = load_yolo(weights)
            self._use_yolo = True

    def predict(self, input_data):
        if self._use_yolo:
            return self.model(input_data, verbose=False, conf=self.CONF_THRESHOLD)

        # SIFT template matching
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_scene, des_scene = self._sift.detectAndCompute(gray, None)
        if des_scene is None:
            return {"matches": []}

        matches_out = []
        for tpl in self._templates:
            raw_matches = self._bf.knnMatch(tpl["des"], des_scene, k=2)
            good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
            if len(good) >= 10:
                src_pts = np.float32([tpl["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    inliers = int(mask.sum())
                    if inliers >= 8:
                        pts = dst_pts[mask.ravel() == 1]
                        x1, y1 = pts.min(axis=0).astype(int).flatten()
                        x2, y2 = pts.max(axis=0).astype(int).flatten()
                        matches_out.append({
                            "name": tpl["name"], "box": (x1, y1, x2, y2),
                            "inliers": inliers, "total_matches": len(good),
                        })
        return {"matches": matches_out}

    def visualize(self, input_data, output):
        if isinstance(output, dict):
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
            vis = frame.copy()
            for m in output.get("matches", []):
                x1, y1, x2, y2 = m["box"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{m['name']} ({m['inliers']})", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            n = len(output.get("matches", []))
            cv2.putText(vis, f"Logos: {n}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            return vis
        return output[0].plot()
