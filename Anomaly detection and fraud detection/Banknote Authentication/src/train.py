#!/usr/bin/env python3
"""
Training Script for: Banknote authentication using the Banknote Authentication dataset
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def train(model, train_data, val_data, config: dict):
    """Train the model."""
    logger.info("Starting training...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Implement training loop here
    raise NotImplementedError("Implement training for this project")
