"""Gesture debouncing for Gesture Controlled Slideshow.

Prevents spurious actions by requiring:
1. The same gesture to be detected for *stable_frames*
   consecutive frames.
2. A minimum *debounce_sec* cooldown between triggered actions.
3. Gesture confidence above *confidence_threshold*.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GestureConfig
from gesture_recognizer import UNKNOWN, GestureState

log = logging.getLogger("gesture.debouncer")


@dataclass
class DebouncedResult:
    """Debounced gesture output."""

    gesture: str = UNKNOWN
    action: str = ""             # mapped slideshow action (or empty)
    triggered: bool = False      # True only when a new action fires
    stable_count: int = 0
    cooldown_remaining: float = 0.0


class GestureDebouncer:
    """Temporal debouncing and action mapping for gestures."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._prev_gesture: str = UNKNOWN
        self._stable_count: int = 0
        self._last_trigger_time: float = 0.0
        self._last_triggered_gesture: str = UNKNOWN

    def update(self, state: GestureState) -> DebouncedResult:
        """Process a new gesture observation.

        Parameters
        ----------
        state : GestureState
            Current frame's recognised gesture.

        Returns
        -------
        DebouncedResult
        """
        result = DebouncedResult(gesture=state.gesture)

        # Confidence gate
        if state.confidence < self.cfg.confidence_threshold:
            self._stable_count = 0
            self._prev_gesture = UNKNOWN
            return result

        # Stability tracking
        if state.gesture == self._prev_gesture:
            self._stable_count += 1
        else:
            self._stable_count = 1
            self._prev_gesture = state.gesture

        result.stable_count = self._stable_count

        # Check if we should trigger
        now = time.monotonic()
        cooldown_left = max(
            0.0,
            self.cfg.debounce_sec - (now - self._last_trigger_time),
        )
        result.cooldown_remaining = cooldown_left

        if (
            state.gesture != UNKNOWN
            and self._stable_count >= self.cfg.stable_frames
            and cooldown_left <= 0
            and state.gesture != self._last_triggered_gesture
        ):
            action = self.cfg.gesture_map.get(state.gesture, "")
            if action:
                result.action = action
                result.triggered = True
                self._last_trigger_time = now
                self._last_triggered_gesture = state.gesture
                log.debug(
                    "Triggered: %s → %s (stable=%d)",
                    state.gesture, action, self._stable_count,
                )

        # Reset last-triggered when gesture changes
        if state.gesture != self._last_triggered_gesture:
            if self._stable_count >= self.cfg.stable_frames:
                self._last_triggered_gesture = UNKNOWN

        return result

    def reset(self) -> None:
        """Clear debouncing state."""
        self._prev_gesture = UNKNOWN
        self._stable_count = 0
        self._last_trigger_time = 0.0
        self._last_triggered_gesture = UNKNOWN
