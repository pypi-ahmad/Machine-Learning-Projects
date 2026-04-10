"""
Central path resolver — single source of truth for all project paths.

Usage:
    from utils.paths import REPO_ROOT, DATA_DIR, MODELS_DIR
    img_path = DATA_DIR / "my_dataset" / "image.jpg"
"""

from __future__ import annotations

from pathlib import Path

# ── Repository root (parent of this file's directory) ──────
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Shared directories ─────────────────────────────────────
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"
SCRIPTS_DIR = REPO_ROOT / "scripts"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# ── Project source directories ─────────────────────────────
PROJECTS_DIR = REPO_ROOT  # projects live at repo root level


def get_project_dir(project_name: str) -> Path:
    """Resolve the *Source Code* directory for a named project.

    Handles the known ``Souce Code`` typo variant automatically.

    Parameters
    ----------
    project_name : str
        Exact top-level folder name, e.g. ``"Brain Tumour Detection"``.

    Returns
    -------
    pathlib.Path
        Resolved path to the project's source directory.

    Raises
    ------
    FileNotFoundError
        If neither ``Source Code`` nor ``Souce Code`` subfolder exists.
    """
    base = PROJECTS_DIR / project_name

    for variant in ("Source Code", "Souce Code", "wildlife image classification"):
        candidate = base / variant
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"No source directory found under '{base}'. "
        f"Looked for: 'Source Code', 'Souce Code', 'wildlife image classification'"
    )


class PathResolver:
    """Convenience helper that resolves centralised model / data / project paths.

    Usage::

        from utils.paths import PathResolver
        paths = PathResolver()
        model_path  = paths.models("age_gender_recognition") / "age_net.caffemodel"
        video_path  = paths.data("pedestrian_detection") / "vid.mp4"
        project_src = paths.project_src("Age Gender Recognition")
    """

    def __init__(self) -> None:
        self.root = REPO_ROOT

    # ── centralised asset directories ──────────────────────
    def models(self, project_slug: str) -> Path:
        """Return ``<repo>/models/<project_slug>/``."""
        return MODELS_DIR / project_slug

    def data(self, project_slug: str) -> Path:
        """Return ``<repo>/data/<project_slug>/``."""
        return DATA_DIR / project_slug

    # ── project source directory ───────────────────────────
    def project_src(self, project_name: str) -> Path:
        """Alias for :func:`get_project_dir`."""
        return get_project_dir(project_name)

    # ── legacy directory ───────────────────────────────────
    def legacy(self, subpath: str = "") -> Path:
        """Return ``<repo>/legacy/<subpath>``."""
        return REPO_ROOT / "legacy" / subpath


# ── Singleton instance for easy import ─────────────────────
_resolver = PathResolver()


def ensure_dirs() -> None:
    """Create all shared directories if they don't exist."""
    for d in (CONFIGS_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR, SCRIPTS_DIR, NOTEBOOKS_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"REPO_ROOT   : {REPO_ROOT}")
    print(f"CONFIGS_DIR : {CONFIGS_DIR}")
    print(f"DATA_DIR    : {DATA_DIR}")
    print(f"MODELS_DIR  : {MODELS_DIR}")
    print(f"LOGS_DIR    : {LOGS_DIR}")
    print(f"SCRIPTS_DIR : {SCRIPTS_DIR}")
    print(f"NOTEBOOKS_DIR: {NOTEBOOKS_DIR}")
    ensure_dirs()
    print("\nAll shared directories verified/created.")
