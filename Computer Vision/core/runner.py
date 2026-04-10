"""Unified runner — single entry-point for any registered project.

Usage::

    from core.runner import discover_projects, run, benchmark

    discover_projects()
    output = run("pedestrian_detection", frame)
    stats  = benchmark("pedestrian_detection", frame, n_runs=20)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from core.registry import PROJECT_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parent.parent
_discovered: set[str] = set()


# ── auto-discovery ─────────────────────────────────────────
def discover_projects() -> int:
    """Import every ``modern.py`` in the workspace to populate the registry.

    Returns the number of newly-loaded modules.
    """
    count = 0
    # Cover both correct and typo variants of "Source Code" / "Souce Code",
    # plus the wildlife project's non-standard layout.
    for pattern in (
        "*/Source Code/modern.py",
        "*/Souce Code/modern.py",
        "*/*/modern.py",
    ):
        for modern_py in sorted(_REPO_ROOT.glob(pattern)):
            mod_key = str(modern_py)
            if mod_key in _discovered:
                continue
            module_name = (
                "_modern_"
                + modern_py.parent.parent.name.replace(" ", "_")
                .replace("-", "_")
                .replace("&", "")
                .lower()
            )
            try:
                spec = importlib.util.spec_from_file_location(module_name, modern_py)
                mod = importlib.util.module_from_spec(spec)
                _discovered.add(mod_key)  # mark before exec to avoid retries on failure
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                count += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to load {modern_py.relative_to(_REPO_ROOT)}: {exc}")
    return count


# ── run / benchmark ────────────────────────────────────────
def run(project_name: str, input_data: Any, *, visualize: bool = False) -> Any:
    """Instantiate, load, and run inference for *project_name*."""
    if project_name not in PROJECT_REGISTRY:
        available = ", ".join(sorted(PROJECT_REGISTRY.keys()))
        raise KeyError(
            f"Unknown project '{project_name}'. Available: {available}"
        )

    project = PROJECT_REGISTRY[project_name]()
    project.load()
    project._loaded = True
    output = project.predict(input_data)

    if visualize:
        return project.visualize(input_data, output)
    return output


def benchmark(project_name: str, input_data: Any, *, n_runs: int = 10) -> dict:
    """Instantiate *project_name* and return benchmark statistics."""
    if project_name not in PROJECT_REGISTRY:
        raise KeyError(f"Unknown project '{project_name}'.")
    project = PROJECT_REGISTRY[project_name]()
    return project.benchmark(input_data, n_runs=n_runs)
