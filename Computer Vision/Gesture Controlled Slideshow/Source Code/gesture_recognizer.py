"""Hand gesture recognition from landmark positions.

Classifies hand poses into named gestures by analysing which
fingers are extended.  The gesture vocabulary is intentionally
small and reliable:

    OPEN_PALM   — all 5 fingers extended
    FIST        — no fingers extended
    POINTING    — only index finger extended
    PEACE       — index + middle extended
    THUMBS_UP   — only thumb extended, hand roughly upright

Finger-up detection uses the standard MediaPipe approach:
    - Fingers 2–5: tip.y < pip.y (in image coords, lower y = higher)
    - Thumb: tip.x vs IP.x, direction depends on handedness
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GestureConfig
from hand_detector import (
    FINGER_PIPS,
    FINGER_TIPS,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandResult,
)

log = logging.getLogger("gesture.recognizer")

# Named gestures
OPEN_PALM = "OPEN_PALM"
FIST = "FIST"
POINTING = "POINTING"
PEACE = "PEACE"
THUMBS_UP = "THUMBS_UP"
UNKNOWN = "UNKNOWN"


@dataclass
class GestureState:
    """Recognised gesture for one frame."""

    gesture: str = UNKNOWN
    fingers_up: list[bool] = field(default_factory=lambda: [False] * 5)
    finger_count: int = 0
    confidence: float = 0.0


class GestureRecognizer:
    """Classify hand landmarks into named gestures."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg

    def recognize(self, hand: HandResult) -> GestureState:
        """Classify the current hand pose.

        Parameters
        ----------
        hand : HandResult
            Detected hand landmarks.

        Returns
        -------
        GestureState
        """
        state = GestureState()
        if not hand.detected:
            return state

        state.confidence = hand.score

        # Determine which fingers are up
        fingers = self._detect_fingers(hand)
        state.fingers_up = fingers
        state.finger_count = sum(fingers)

        # Classify gesture
        thumb, index, middle, ring, pinky = fingers

        if index and middle and ring and pinky:
            state.gesture = OPEN_PALM
        elif not any(fingers):
            state.gesture = FIST
        elif index and not middle and not ring and not pinky and not thumb:
            state.gesture = POINTING
        elif index and middle and not ring and not pinky:
            state.gesture = PEACE
        elif thumb and not index and not middle and not ring and not pinky:
            # Thumb up — verify hand is roughly upright
            if self._is_thumb_up_pose(hand):
                state.gesture = THUMBS_UP
            else:
                state.gesture = UNKNOWN
        else:
            state.gesture = UNKNOWN

        return state

    def _detect_fingers(self, hand: HandResult) -> list[bool]:
        """Detect which fingers are extended.

        Returns
        -------
        list[bool]
            [thumb, index, middle, ring, pinky]
        """
        lm = hand.landmarks

        # Thumb: compare tip.x vs IP.x
        # For right hand (in image): tip.x > ip.x means extended
        # For left hand (in image): tip.x < ip.x means extended
        # MediaPipe handedness is from camera's perspective (mirrored)
        thumb_tip = lm[THUMB_TIP]
        thumb_ip = lm[THUMB_IP]

        if hand.handedness == "Right":
            # Camera right = user's left (mirrored)
            thumb_up = thumb_tip.x < thumb_ip.x
        else:
            thumb_up = thumb_tip.x > thumb_ip.x

        # Fingers 2–5: tip.y < pip.y means extended
        # (lower y = higher in image)
        fingers = [thumb_up]
        for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
            tip_y = lm[tip_idx].y
            pip_y = lm[pip_idx].y
            fingers.append(tip_y < pip_y - self.cfg.finger_up_margin)

        return fingers

    def _is_thumb_up_pose(self, hand: HandResult) -> bool:
        """Check if the hand is in a thumbs-up orientation.

        The wrist should be below the thumb MCP, and fingers
        should be curled.
        """
        lm = hand.landmarks
        wrist_y = lm[WRIST].y
        thumb_mcp_y = lm[THUMB_MCP].y
        # Wrist below thumb MCP (higher y value)
        return wrist_y > thumb_mcp_y
