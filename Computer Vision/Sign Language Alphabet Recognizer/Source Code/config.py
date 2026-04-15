"""Sign Language Alphabet Recognizer -- configuration dataclass."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


# ASL static alphabet: A-Z minus J and Z (which require motion)
ASL_STATIC_LABELS: list[str] = [
    c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"
]


@dataclasses.dataclass
class SignLangConfig:
    """All tunables for the sign-language recognition pipeline."""

    # --- MediaPipe Hand Landmarker ---
    max_num_hands: int = 1
    model_complexity: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False

    # --- Feature extraction ---
    normalise_to_wrist: bool = True   # translate landmarks so wrist = origin
    scale_invariant: bool = True      # scale so max distance from wrist = 1

    # --- Classifier ---
    model_path: str = "model/sign_lang_clf.pkl"
    labels: list[str] = dataclasses.field(default_factory=lambda: list(ASL_STATIC_LABELS))

    # --- Smoothing ---
    enable_smoothing: bool = True
    vote_window: int = 5

    # --- Display ---
    show_landmarks: bool = True
    show_prediction: bool = True
    show_confidence: bool = True

    # --- Output ---
    output_dir: str = "output"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SignLangConfig:
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


def load_config(path: str | Path) -> SignLangConfig:
    """Load config from YAML or JSON, falling back to defaults."""
    p = Path(path)
    if not p.exists():
        return SignLangConfig()
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(text) or {}
        except Exception:
            data = {}
    else:
        data = json.loads(text)
    return SignLangConfig.from_dict(data)
