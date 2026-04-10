"""Interactive Video Object Cutout Studio — video mask propagation.

Wraps SAM2VideoPredictor for propagating mask prompts across video
frames.  Handles frame extraction from video files and state
management for the video predictor.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import cv2
import numpy as np

if TYPE_CHECKING:
    from config import CutoutConfig


@dataclass
class PropagationResult:
    """Masks propagated across all frames of a video."""

    frame_masks: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    frame_count: int = 0

    def get_mask(self, frame_idx: int, obj_id: int = 1) -> np.ndarray | None:
        obj_masks = self.frame_masks.get(frame_idx)
        if obj_masks is None:
            return None
        return obj_masks.get(obj_id)


class VideoPropagator:
    """SAM 2 video predictor — propagates prompts across frames."""

    def __init__(self, cfg: CutoutConfig | None = None) -> None:
        if cfg is None:
            from config import CutoutConfig
            cfg = CutoutConfig()
        self.cfg = cfg
        self._predictor = None
        self._device = None

    # ── lifecycle ──────────────────────────────────────────

    def load(self) -> None:
        """Load SAM 2 video predictor (auto-downloads weights from HF)."""
        import torch
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        device = self.cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self._predictor = SAM2VideoPredictor.from_pretrained(
            self.cfg.model_id, device=self._device,
        )

    def close(self) -> None:
        self._predictor = None

    # ── frame extraction ──────────────────────────────────

    def extract_frames(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> tuple[Path, int]:
        """Extract JPEG frames from *video_path* into *output_dir*.

        Returns (frames_dir, frame_count).
        """
        video_path = Path(video_path)
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        count = 0
        frame_idx = 0
        stride = max(1, self.cfg.frame_stride)
        max_frames = self.cfg.max_frames

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                out_path = output_dir / f"{count:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                count += 1
                if max_frames > 0 and count >= max_frames:
                    break
            frame_idx += 1

        cap.release()
        return output_dir, count

    # ── video prediction ──────────────────────────────────

    def init_video(self, frames_dir: str | Path) -> object:
        """Initialise SAM 2 inference state for a directory of JPEG frames."""
        import torch

        if self._predictor is None:
            self.load()

        ac_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        with torch.inference_mode(), torch.autocast(
            self._device.type, dtype=ac_dtype,
        ):
            state = self._predictor.init_state(video_path=str(frames_dir))
        return state

    def add_prompt(
        self,
        state: object,
        frame_idx: int,
        obj_id: int,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
    ) -> np.ndarray:
        """Add point/box prompt on *frame_idx*, returns initial mask."""
        import torch

        if self._predictor is None:
            raise RuntimeError("Call load() or init_video() first.")

        ac_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        with torch.inference_mode(), torch.autocast(
            self._device.type, dtype=ac_dtype,
        ):
            _, _, masks = self._predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                box=box,
            )

        # masks shape: (num_obj, 1, H, W) → squeeze to (H, W)
        mask = (masks[0, 0] > 0.0).cpu().numpy()
        return mask

    def propagate(self, state: object) -> PropagationResult:
        """Propagate prompts across all frames.  Returns aggregated masks."""
        import torch

        if self._predictor is None:
            raise RuntimeError("Call load() or init_video() first.")

        result = PropagationResult()
        ac_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32

        with torch.inference_mode(), torch.autocast(
            self._device.type, dtype=ac_dtype,
        ):
            for frame_idx, obj_ids, masks in self._predictor.propagate_in_video(state):
                obj_masks: dict[int, np.ndarray] = {}
                for oid, mask in zip(obj_ids, masks):
                    obj_masks[int(oid)] = (mask[0] > 0.0).cpu().numpy()
                result.frame_masks[int(frame_idx)] = obj_masks
                result.frame_count = max(result.frame_count, int(frame_idx) + 1)

        return result
