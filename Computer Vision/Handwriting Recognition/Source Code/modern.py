"""Modern v2 pipeline — Handwriting Recognition.

Replaces: Custom TF/Keras HTR model
Uses:     TrOCR (HuggingFace) for handwritten text line/word recognition,
          with PaddleOCR as fallback OCR stack.

Pipeline: preprocess image → TrOCR generates text autoregressively
          (no image-level classification — proper sequence-to-sequence OCR)

The original implementation is preserved in src/main.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register


@register("handwriting_recognition")
class HandwritingRecognitionModern(CVProject):
    project_type = "classification"
    description = "Handwritten text recognition (sequence-to-sequence OCR)"
    legacy_tech = "Custom TF/Keras HTR CNN+RNN"
    modern_tech = "TrOCR (HuggingFace) / PaddleOCR"

    _backend = None  # "trocr", "paddle", or None
    _processor = None
    _model = None
    _ocr = None

    def load(self):
        # Priority 1: TrOCR (best for handwritten text)
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            model_name = "microsoft/trocr-base-handwritten"
            self._processor = TrOCRProcessor.from_pretrained(model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(device)
            self._model.eval()
            self._backend = "trocr"
            print(f"  [handwriting] Using TrOCR ({model_name}) on {device}")
            return
        except ImportError:
            pass

        # Priority 2: PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)
            self._backend = "paddle"
            print("  [handwriting] TrOCR not available — using PaddleOCR")
            return
        except ImportError:
            pass

        print("  [handwriting] No OCR engine available")
        print("  [handwriting] Install: pip install transformers  (for TrOCR)")
        print("  [handwriting]      or: pip install paddleocr     (for PaddleOCR)")

    def predict(self, input_data):
        if isinstance(input_data, (str, Path)):
            image = cv2.imread(str(input_data))
        else:
            image = input_data

        if self._backend == "trocr":
            import torch
            from PIL import Image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pixel_values = self._processor(images=pil_img, return_tensors="pt").pixel_values
            device = next(self._model.parameters()).device
            pixel_values = pixel_values.to(device)
            with torch.no_grad():
                generated_ids = self._model.generate(pixel_values, max_new_tokens=128)
            text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return {"text": text, "backend": "TrOCR"}

        if self._backend == "paddle":
            result = self._ocr.ocr(image, cls=True)
            lines = []
            if result and result[0]:
                for line in result[0]:
                    pts = np.array(line[0], dtype=np.int32)
                    x1, y1 = pts.min(axis=0)
                    x2, y2 = pts.max(axis=0)
                    text, conf = line[1]
                    lines.append({
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "text": text,
                        "conf": float(conf),
                    })
            full_text = " ".join(l["text"] for l in lines)
            return {"text": full_text, "lines": lines, "backend": "PaddleOCR"}

        return {"text": "", "error": "No OCR engine installed", "backend": None}

    def visualize(self, input_data, output):
        if isinstance(input_data, (str, Path)):
            vis = cv2.imread(str(input_data))
        else:
            vis = input_data.copy()

        text = output.get("text", "")
        backend = output.get("backend", "none")

        # Draw recognized text lines if available
        for line in output.get("lines", []):
            x1, y1, x2, y2 = line["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Overlay recognized text at top
        label = f"[{backend}] {text[:80]}" if text else f"[{backend}] (no text)"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return vis
