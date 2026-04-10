"""Core unified inference engine — Phase 3A modernization.

Usage::

    from core import run, benchmark, list_projects, discover_projects

    discover_projects()          # auto-import all modern.py wrappers
    print(list_projects())       # show registered project names
    output = run("pedestrian_detection", frame)
    stats  = benchmark("pedestrian_detection", frame, n_runs=20)
"""

from core.base import CVProject
from core.registry import PROJECT_REGISTRY, register, list_projects
from core.runner import run, benchmark, discover_projects

__all__ = [
    "CVProject",
    "PROJECT_REGISTRY",
    "register",
    "list_projects",
    "run",
    "benchmark",
    "discover_projects",
]
