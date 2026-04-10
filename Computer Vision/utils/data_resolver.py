"""Safe dataset resolver — finds data locally, NEVER auto-downloads blindly.

Usage::

    from utils.data_resolver import resolve_data

    video = resolve_data("pedestrian_detection", "vid.mp4")
    folder = resolve_data("emotion_recognition")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.paths import REPO_ROOT, DATA_DIR


def resolve_data(
    project_slug: str,
    filename: Optional[str] = None,
    project_src: Optional[Path] = None,
) -> Path:
    """Resolve a dataset file/directory by checking local paths.

    Search order
    ------------
    1. ``<repo>/data/<project_slug>/[filename]``
    2. ``<project_src>/[filename]``
    3. ``<project_src>/data/[filename]``
    4. ``<project_src>/Data/[filename]``

    Raises
    ------
    RuntimeError
        If the dataset cannot be found at any candidate location.
    """
    candidates: list[Path] = []

    # 1. Centralised data directory
    base = DATA_DIR / project_slug
    candidates.append(base / filename if filename else base)

    # 2–4. Project-local directories
    if project_src is not None:
        if filename:
            candidates.append(project_src / filename)
            candidates.append(project_src / "data" / filename)
            candidates.append(project_src / "Data" / filename)
        else:
            candidates.append(project_src)
            candidates.append(project_src / "data")
            candidates.append(project_src / "Data")

    for c in candidates:
        if c.exists():
            return c

    raise RuntimeError(
        f"Dataset not found for '{project_slug}'"
        + (f" (file: {filename})" if filename else "")
        + ".\nChecked:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\n\nPlease provide the dataset path or URL in project_meta.yaml."
    )
