"""
Root conftest.py — shared fixtures for all NLP project tests.
"""
import pytest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = WORKSPACE_ROOT / "data"


@pytest.fixture(scope="session")
def workspace_root():
    """Absolute path to workspace root."""
    return WORKSPACE_ROOT


@pytest.fixture(scope="session")
def data_root():
    """Absolute path to centralized data/ directory."""
    return DATA_ROOT


def _make_data_dir_fixture(slug):
    """Factory: create a fixture that returns data/<slug> path."""
    @pytest.fixture(scope="module")
    def data_dir():
        d = DATA_ROOT / slug
        if not d.exists():
            pytest.skip(f"Data directory missing: {d}")
        return d
    return data_dir
