"""TrOCR engine wrapper for handwritten text recognition.

Wraps HuggingFace ``VisionEncoderDecoderModel`` + ``TrOCRProcessor``
with lazy initialisation and per-line confidence scores.

Usage::

    from ocr_engine import TrOCREngine
    from config import NoteConfig

    engine = TrOCREngine(NoteConfig())
    result = engine.recognise(line_crop)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("handwritten_note.ocr_engine")


@dataclass
class RecognitionResult:
    """Result for a single line image."""

    text: str
    confidence: float              # mean token-level log-prob → [0, 1]
    token_scores: list[float]      # per-token confidence


class TrOCREngine:
    """TrOCR wrapper with lazy model loading."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._processor = None
        self._model = None

    def _init_model(self) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        log.info("Loading TrOCR model: %s", self.cfg.model_name)
        self._processor = TrOCRProcessor.from_pretrained(self.cfg.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self.cfg.model_name,
        )

        if self.cfg.use_gpu:
            import torch
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
                log.info("TrOCR model moved to CUDA")
            else:
                log.warning("GPU requested but CUDA not available")

        self._model.eval()
        log.info("TrOCR model loaded successfully")

    def recognise(self, image: np.ndarray) -> RecognitionResult:
        """Recognise handwritten text in a single line/crop image.

        Parameters
        ----------
        image : np.ndarray
            BGR or RGB image of a single text line.

        Returns
        -------
        RecognitionResult
            Recognised text with confidence scores.
        """
        if self._model is None:
            self._init_model()

        import torch
        from PIL import Image

        # Convert BGR → RGB → PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        pil_img = Image.fromarray(rgb).convert("RGB")

        pixel_values = self._processor(
            images=pil_img, return_tensors="pt",
        ).pixel_values

        device = next(self._model.parameters()).device
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=self.cfg.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode text
        ids = outputs.sequences[0]
        text = self._processor.batch_decode(
            [ids], skip_special_tokens=True,
        )[0].strip()

        # Compute per-token confidence
        token_scores = self._compute_confidence(outputs)
        mean_conf = (
            float(np.mean(token_scores)) if token_scores else 0.0
        )

        return RecognitionResult(
            text=text,
            confidence=mean_conf,
            token_scores=token_scores,
        )

    def recognise_batch(
        self, images: list[np.ndarray],
    ) -> list[RecognitionResult]:
        """Recognise multiple line images (sequential for simplicity)."""
        return [self.recognise(img) for img in images]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(outputs) -> list[float]:
        """Extract per-token confidence from beam search outputs."""
        import torch

        if not hasattr(outputs, "scores") or outputs.scores is None:
            return []

        scores: list[float] = []
        for step_scores in outputs.scores:
            probs = torch.softmax(step_scores[0], dim=-1)
            max_prob = float(probs.max())
            scores.append(max_prob)

        return scores
