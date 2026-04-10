"""Modern v2 pipeline — Medical Image Segmentation for Tumour Detection.

Replaces: Notebook-only brain tumour segmentation
Uses:     YOLO26m-seg for medical-image segmentation (primary, trainable)
          Optional MedSAM comparison track (set MEDSAM_COMPARE=1)

Pipeline: YOLO26m-seg → tumour region masks
Compare:  MedSAM side-by-side evaluation (if installed + checkpoint)

Fine-tune: python train_segmentation.py --data tumour_dataset.yaml

The original notebook implementation is preserved in the .ipynb file.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("medical_image_segmentation")
class MedicalImageSegmentationModern(CVProject):
    project_type = "segmentation"
    description = "Brain tumour segmentation via YOLO-seg"
    legacy_tech = "Custom segmentation (notebook)"
    modern_tech = "YOLO26m-seg (trainable; optional MedSAM comparison)"

    _yolo = None
    _mask_generator = None
    _backend = "none"

    def load(self):
        # Primary: YOLO-seg (trainable — fine-tune on tumour data for best results)
        from utils.yolo import load_yolo
        from models.registry import resolve
        weights, ver, fallback = resolve("medical_image_segmentation", "medical_seg")
        self._yolo = load_yolo(weights)
        self._backend = "yolo"
        print(f"  [medical_seg] YOLO-seg loaded ({weights})")
        if fallback:
            print("  [medical_seg] Fine-tune: python train_segmentation.py --data tumour.yaml")
        # Optional MedSAM comparison track (set MEDSAM_COMPARE=1 to enable)
        if os.environ.get("MEDSAM_COMPARE"):
            self._load_medsam_comparison()

    def _load_medsam_comparison(self):
        """Load MedSAM alongside YOLO for side-by-side comparison."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            ckpt = self._find_checkpoint("medsam_vit_b.pth")
            if ckpt is None:
                ckpt = self._find_checkpoint("sam_vit_b_01ec64.pth")
            if ckpt is not None:
                sam = sam_model_registry["vit_b"](checkpoint=str(ckpt))
                self._mask_generator = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=32,
                    pred_iou_thresh=0.88,
                    stability_score_thresh=0.92,
                    min_mask_region_area=500,
                )
                self._backend = "medsam"
                print(f"  [medical_seg] MedSAM comparison loaded ({ckpt.name})")
        except ImportError:
            print("  [medical_seg] MedSAM not installed — comparison skipped")

    @staticmethod
    def _find_checkpoint(name: str):
        candidates = [
            Path(__file__).resolve().parents[2] / "models" / "medical_image_segmentation" / name,
            Path(__file__).resolve().parents[2] / "models" / name,
            Path.home() / ".cache" / "medsam" / name,
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def predict(self, input_data):
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))

        if self._backend == "medsam":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks = self._mask_generator.generate(rgb)
            masks = sorted(masks, key=lambda m: m["area"], reverse=True)
            return {"masks": masks, "backend": "medsam"}

        return self._yolo(frame, verbose=False)

    def visualize(self, input_data, output):
        if isinstance(output, dict) and "masks" in output:
            frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
            vis = frame.copy()
            for i, m in enumerate(output["masks"]):
                mask = m["segmentation"]
                color = np.random.RandomState(i).randint(0, 255, 3).tolist()
                vis[mask] = (vis[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            n = len(output["masks"])
            cv2.putText(vis, f"Regions: {n} [MedSAM]", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return vis
        return output[0].plot()
