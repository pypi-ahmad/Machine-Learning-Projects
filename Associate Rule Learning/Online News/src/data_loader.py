#!/usr/bin/env python3
"""
Data Loader for: Online News
Handles dataset loading, splitting, and DataLoader creation.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(data_dir: Path = DATA_DIR):
    """Load the dataset from data directory."""
    logger.info("Loading data from %s", data_dir)
    # Implement dataset-specific loading logic here
    raise NotImplementedError("Implement dataset loading for this project")


def get_splits(X, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split data into train/val/test sets."""
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=seed, stratify=y if y is not None else None
    )
    val_fraction = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_fraction), random_state=seed
    )
    logger.info("Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test
