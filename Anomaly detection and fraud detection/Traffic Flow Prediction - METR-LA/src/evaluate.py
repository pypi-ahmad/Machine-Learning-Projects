#!/usr/bin/env python3
"""
Evaluation Script for: Traffic flow prediction using the METR-LA traffic
Problem Type: Regression (Time Series)
Metrics: rmse, mae, r2
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def evaluate(model, test_data, test_labels):
    """Evaluate model performance."""
    logger.info("Evaluating model...")
    # Implement evaluation with appropriate metrics:
    # rmse, mae, r2
    raise NotImplementedError("Implement evaluation for this project")
