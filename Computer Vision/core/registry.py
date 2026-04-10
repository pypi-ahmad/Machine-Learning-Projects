"""Global registry for modernized CV project wrappers.

Use the :func:`register` decorator inside each ``modern.py``::

    from core.registry import register

    @register("my_project")
    class MyProject(CVProject):
        ...
"""

from __future__ import annotations

from typing import Dict, Type

from core.base import CVProject

PROJECT_REGISTRY: Dict[str, Type[CVProject]] = {}


def register(name: str):
    """Class decorator — registers *cls* under *name* in the global registry."""
    def wrapper(cls: Type[CVProject]):
        if name in PROJECT_REGISTRY:
            raise ValueError(f"Project '{name}' is already registered.")
        PROJECT_REGISTRY[name] = cls
        return cls
    return wrapper


def list_projects() -> list[str]:
    """Return sorted list of all registered project names."""
    return sorted(PROJECT_REGISTRY.keys())
