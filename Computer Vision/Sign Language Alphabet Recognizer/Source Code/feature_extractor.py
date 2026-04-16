"""Sign Language Alphabet Recognizer -- landmark -> feature vector.

Converts 21 hand landmarks into a normalised, translation- and
scale-invariant feature vector suitable for classification.

The feature vector combines:
- 63 coordinates: (x, y, z) for each of the 21 landmarks
- 5 fingertip-to-wrist distances
- 5 fingertip-to-joint distances
- 4 adjacent fingertip spread distances
"""

from __future__ import annotations

import numpy as np

from hand_detector import NUM_LANDMARKS, WRIST, HandResult

FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)
REFERENCE_JOINT_INDICES = np.array([3, 6, 10, 14, 18], dtype=np.int64)


def extract_features(
    hand: HandResult,
    normalise_to_wrist: bool = True,
    scale_invariant: bool = True,
) -> np.ndarray:
    """Return a normalised float32 feature vector from *hand* landmarks.

    Steps:
    1. Collect (x, y, z) for all 21 landmarks.
    2. Optionally translate so wrist = (0, 0).
    3. Optionally scale so max Euclidean distance from wrist = 1.
    4. Add compact geometric descriptors for fingertip shape.
    """
    coords = np.array(
        [(hand.landmarks[i].x, hand.landmarks[i].y, hand.landmarks[i].z) for i in range(NUM_LANDMARKS)],
        dtype=np.float32,
    )
    return extract_features_from_landmarks_raw(coords, normalise_to_wrist, scale_invariant)


def extract_features_from_landmarks_raw(
    landmarks_xy: np.ndarray,
    normalise_to_wrist: bool = True,
    scale_invariant: bool = True,
) -> np.ndarray:
    """Same as :func:`extract_features` but takes a (21, 2|3) array directly.

    Useful for unit testing without a full HandResult object.
    """
    coords = landmarks_xy.astype(np.float32).copy()
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)])
    if normalise_to_wrist:
        coords -= coords[WRIST]
    if scale_invariant:
        dists = np.linalg.norm(coords, axis=1)
        max_dist = dists.max()
        if max_dist > 1e-6:
            coords /= max_dist

    xyz_flat = coords.flatten()
    tip_wrist_dists = np.linalg.norm(coords[FINGERTIP_INDICES], axis=1)
    tip_joint_dists = np.linalg.norm(
        coords[FINGERTIP_INDICES] - coords[REFERENCE_JOINT_INDICES],
        axis=1,
    )
    fingertip_spread = np.linalg.norm(
        np.diff(coords[FINGERTIP_INDICES, :2], axis=0),
        axis=1,
    )
    return np.concatenate(
        [xyz_flat, tip_wrist_dists, tip_joint_dists, fingertip_spread]
    ).astype(np.float32)
