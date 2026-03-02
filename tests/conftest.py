"""
Shared pytest fixtures and configuration for ML project tests.
"""
import pytest
from pathlib import Path
import warnings

# Suppress noisy warnings during test runs
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"


@pytest.fixture(scope="session")
def workspace_root():
    """Return the workspace root path."""
    return ROOT


@pytest.fixture(scope="session")
def data_root():
    """Return the centralized data directory."""
    return DATA_ROOT


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that require missing data directories."""
    for item in items:
        # If a test has a 'data' marker and data doesn't exist, skip it
        if "data" in [m.name for m in item.iter_markers()]:
            # Extract DATA_DIR from the test module if available
            module = item.module
            data_dir = getattr(module, "DATA_DIR", None)
            if data_dir and not Path(data_dir).exists():
                item.add_marker(
                    pytest.mark.skip(reason=f"Data directory not found: {data_dir}")
                )
