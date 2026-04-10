"""Visualization overlays for Gesture Controlled Slideshow.

Draws hand landmarks, gesture label, finger state, slide counter,
action banner, and optional pointer cursor onto frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GestureConfig
from controller import ControllerResult
from hand_detector import HandDetector

_FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def draw_overlay(
    frame: np.ndarray,
    result: ControllerResult,
    cfg: GestureConfig,
    *,
    detector: HandDetector | None = None,
) -> np.ndarray:
    """Draw gesture analysis overlays on a frame copy.

    Parameters
    ----------
    frame : np.ndarray
        BGR source frame.
    result : ControllerResult
        Pipeline output for this frame.
    cfg : GestureConfig
        Display settings.
    detector : HandDetector, optional
        If provided, draws full MediaPipe hand skeleton.

    Returns
    -------
    np.ndarray
        Annotated copy.
    """
    vis = frame.copy()

    # Hand landmarks
    if cfg.show_hand_landmarks and result.hand_detected and detector:
        detector.draw_landmarks(vis, result.hand)

    # Gesture label
    if cfg.show_gesture_label:
        gesture = result.gesture.gesture if result.hand_detected else "---"
        color = (0, 255, 0) if result.hand_detected else (100, 100, 100)
        cv2.putText(
            vis, f"Gesture: {gesture}", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

    # Finger state
    if cfg.show_finger_state and result.hand_detected:
        fingers = result.gesture.fingers_up
        parts = []
        for name, up in zip(_FINGER_NAMES, fingers):
            parts.append(f"{name[0]}:{'1' if up else '0'}")
        finger_str = " ".join(parts)
        cv2.putText(
            vis, finger_str, (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

    # Action banner
    if cfg.show_action_banner and result.debounced.triggered:
        _draw_action_banner(vis, result.debounced.action)

    # Slide counter
    if cfg.show_slide_counter:
        slide = result.slide
        counter = f"Slide {slide.current_index + 1}/{slide.total_slides}"
        if slide.paused:
            counter += " [PAUSED]"
        if slide.pointer_mode:
            counter += " [PTR]"

        h = vis.shape[0]
        cv2.putText(
            vis, counter, (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

    # Pointer cursor (index fingertip)
    if result.slide.pointer_mode and result.hand_detected:
        tip_x, tip_y = result.hand.pixel_coords(8)  # INDEX_TIP
        cv2.circle(vis, (int(tip_x), int(tip_y)), 10, (0, 0, 255), 2)
        cv2.circle(vis, (int(tip_x), int(tip_y)), 3, (0, 0, 255), -1)

    return vis


def draw_slide_with_cam(
    slide: np.ndarray,
    cam_frame: np.ndarray,
    result: ControllerResult,
    cfg: GestureConfig,
    *,
    detector: HandDetector | None = None,
    cam_scale: float = 0.3,
) -> np.ndarray:
    """Compose slide image with camera PiP overlay.

    Parameters
    ----------
    slide : np.ndarray
        Current slide image.
    cam_frame : np.ndarray
        Webcam frame.
    result : ControllerResult
        Current pipeline result.
    cfg : GestureConfig
        Display settings.
    detector : HandDetector, optional
        For drawing hand landmarks on the camera view.
    cam_scale : float
        Scale factor for the camera PiP window.

    Returns
    -------
    np.ndarray
        Composed display frame.
    """
    display = slide.copy()
    dh, dw = display.shape[:2]

    # Annotate camera frame
    cam_vis = draw_overlay(cam_frame, result, cfg, detector=detector)

    # Resize camera for PiP
    ch = int(cam_vis.shape[0] * cam_scale)
    cw = int(cam_vis.shape[1] * cam_scale)
    if ch > 0 and cw > 0:
        pip = cv2.resize(cam_vis, (cw, ch))

        # Place in bottom-right corner
        y1 = max(0, dh - ch - 10)
        x1 = max(0, dw - cw - 10)
        y2, x2 = y1 + ch, x1 + cw

        if y2 <= dh and x2 <= dw:
            display[y1:y2, x1:x2] = pip
            cv2.rectangle(
                display, (x1, y1), (x2, y2), (200, 200, 200), 1,
            )

    # Slide counter on display
    slide_state = result.slide
    counter = f"{slide_state.current_index + 1}/{slide_state.total_slides}"
    if slide_state.paused:
        counter += " PAUSED"
    cv2.putText(
        display, counter, (10, dh - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    )

    return display


def _draw_action_banner(img: np.ndarray, action: str) -> None:
    """Draw a brief action notification banner."""
    h, w = img.shape[:2]
    banner_h = 50
    y1 = h // 2 - banner_h // 2
    y2 = y1 + banner_h

    overlay = img.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    label = action.upper()
    text_size = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2,
    )[0]
    tx = (w - text_size[0]) // 2
    ty = y1 + (banner_h + text_size[1]) // 2
    cv2.putText(
        img, label, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2,
    )
