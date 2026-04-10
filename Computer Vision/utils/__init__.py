"""
Shared utilities for Computer Vision Projects.
"""

from utils.device import get_device
from utils.paths import REPO_ROOT, CONFIGS_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR, PathResolver
from utils.logger import get_logger
from utils.datasets import DatasetResolver, DATASET_REGISTRY

__all__ = [
    "get_device",
    "REPO_ROOT",
    "CONFIGS_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    "PathResolver",
    "get_logger",
    "DatasetResolver",
    "DATASET_REGISTRY",
]
