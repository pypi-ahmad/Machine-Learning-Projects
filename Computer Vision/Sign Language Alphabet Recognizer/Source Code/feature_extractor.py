"""Sign Language Alphabet Recognizer — landmark → feature vector.

Converts 21 hand landmarks into a normalised, translation- and
scale-invariant feature vector suitable for classification.

The feature vector is 42-dimensional: (x, y) for each of the 21
landmarks, translated so the wrist is at the origin and optionally
scaled so the maximum distance from the wrist is 1.0.
"""

from __future__ import annotations

import math

import numpy as np

from hand_detector import NUM_LANDMARKS, WRIST, HandResult


def extract_features(
    hand: HandResult,
    normalise_to_wrist: bool = True,
    scale_invariant: bool = True,
) -> np.ndarray:
    """Return a flat (42,) float32 feature vector from *hand* landmarks.

    Steps:
    1. Collect (x, y) for all 21 landmarks.
    2. Optionally translate so wrist = (0, 0).
    3. Optionally scale so max Euclidean distance from wrist = 1.
    4. Flatten to a 1-D array.
    """
    coords = np.array(
        [(hand.landmarks[i].x, hand.landmarks[i].y) for i in range(NUM_LANDMARKS)],
        dtype=np.float32,
    )

    if normalise_to_wrist:
        wrist = coords[WRIST].copy()
        coords -= wrist

    if scale_invariant:
        dists = np.linalg.norm(coords, axis=1)
        max_dist = dists.max()
        if max_dist > 1e-6:
            coords /= max_dist

    return coords.flatten()  # shape (42,)


def extract_features_from_landmarks_raw(
    landmarks_xy: np.ndarray,
    normalise_to_wrist: bool = True,
    scale_invariant: bool = True,
) -> np.ndarray:
    """Same as :func:`extract_features` but takes a (21, 2) array directly.

    Useful for unit testing without a full HandResult object.
    """
    coords = landmarks_xy.astype(np.float32).copy()
    if normalise_to_wrist:
        coords -= coords[WRIST]
    if scale_invariant:
        dists = np.linalg.norm(coords, axis=1)
        max_dist = dists.max()
        if max_dist > 1e-6:
            coords /= max_dist
    return coords.flatten()
