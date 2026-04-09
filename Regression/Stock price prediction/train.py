#!/usr/bin/env python3
"""
Model training for Stock price prediction

Auto-generated from: netflix_stock_price_prediction.ipynb
Project: Stock price prediction
Category: Regression | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score
import os
import warnings

# ======================================================================
# TRAINING PIPELINE
# ======================================================================

def main():
    """Run the training pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    df = load_dataset('stock_price_prediction')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    viz = df.copy()



    # --- PREPROCESSING ───────────────────────────────────────

    train, test = train_test_split(df, test_size = 0.2, random_state=42)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    test_pred = test.copy()

    x_train = train[['Open', 'High', 'Low', 'Volume']].values
    x_test = test[['Open', 'High', 'Low', 'Volume']].values

    y_train = train['Close'].values
    y_test = test['Close'].values



    # --- MODEL TRAINING ──────────────────────────────────────

    model_lnr = LinearRegression()
    model_lnr.fit(x_train, y_train)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model training for Stock price prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
