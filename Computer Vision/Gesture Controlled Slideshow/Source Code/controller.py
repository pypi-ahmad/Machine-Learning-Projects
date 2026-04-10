"""Pipeline controller for Gesture Controlled Slideshow.

Orchestrates: hand detection → gesture recognition → debouncing
→ slideshow action.

Usage::

    from controller import SlideshowController

    ctrl = SlideshowController(cfg)
    ctrl.load()
    result = ctrl.process(frame)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GestureConfig
from debouncer import DebouncedResult, GestureDebouncer
from gesture_recognizer import GestureRecognizer, GestureState
from hand_detector import HandDetector, HandResult
from slideshow import SlideState, Slideshow

log = logging.getLogger("gesture.controller")


@dataclass
class ControllerResult:
    """Complete pipeline result for a single frame."""

    hand: HandResult = field(default_factory=HandResult)
    gesture: GestureState = field(default_factory=GestureState)
    debounced: DebouncedResult = field(default_factory=DebouncedResult)
    slide: SlideState = field(default_factory=SlideState)
    hand_detected: bool = False


class SlideshowController:
    """Full gesture-controlled slideshow pipeline.

    hand detection → gesture classify → debounce → slideshow action
    """

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self.detector = HandDetector(cfg)
        self.recognizer = GestureRecognizer(cfg)
        self.debouncer = GestureDebouncer(cfg)
        self.slideshow = Slideshow(cfg)
        self._loaded = False

    def load(self, slide_dir: str | Path | None = None) -> None:
        """Initialize all components."""
        ok = self.detector.load()
        self._loaded = ok

        self.slideshow.load_slides(slide_dir)

        if ok:
            log.info("Controller pipeline ready")
        else:
            log.error("Pipeline failed to load — MediaPipe unavailable")

    def process(self, frame: np.ndarray) -> ControllerResult:
        """Process a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        ControllerResult
        """
        if not self._loaded:
            self.load()

        result = ControllerResult()

        # 1. Detect hand
        hand = self.detector.detect(frame)
        result.hand = hand
        result.hand_detected = hand.detected

        if not hand.detected:
            result.slide = self.slideshow.state
            return result

        # 2. Recognise gesture
        gesture = self.recognizer.recognize(hand)
        result.gesture = gesture

        # 3. Debounce
        debounced = self.debouncer.update(gesture)
        result.debounced = debounced

        # 4. Execute action
        if debounced.triggered and debounced.action:
            self.slideshow.execute(debounced.action)

        result.slide = self.slideshow.state
        return result

    def handle_key(self, key: int) -> str:
        """Handle keyboard input as fallback.

        Parameters
        ----------
        key : int
            cv2.waitKey return value (masked to 0xFF).

        Returns
        -------
        str
            Action executed, or empty string.
        """
        if not self.cfg.enable_keyboard:
            return ""

        action = ""
        if key == self.cfg.key_next:
            action = "next"
        elif key == self.cfg.key_prev:
            action = "previous"
        elif key == self.cfg.key_pause:
            action = "pause"
        elif key == self.cfg.key_pointer:
            action = "pointer"

        if action:
            self.slideshow.execute(action)
        return action

    def reset(self) -> None:
        """Reset all state."""
        self.debouncer.reset()
        self.slideshow.reset()
