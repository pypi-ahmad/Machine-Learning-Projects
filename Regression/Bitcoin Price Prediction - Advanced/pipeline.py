#!/usr/bin/env python3
"""
Full pipeline for Predicting the price of bitcoin

Auto-generated from: bitcoin_price_prediction.ipynb
Project: Predicting the price of bitcoin
Category: Regression | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')
# Additional imports extracted from mixed cells
from pycaret.time_series import *

# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    """Run the complete pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    # Load data
    df = load_dataset('predicting_the_price_of_bitcoin')

    # Derive Weighted_Price as (High + Low + Close) / 3 (VWAP approximation)
    # Original dataset had a Weighted_Price column; substitute dataset does not
    if 'Weighted_Price' not in df.columns:
        df['Weighted_Price'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Validation
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    df.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Unix-time to 
    df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')

    # Resampling to daily frequency
    df.index = df.Timestamp
    df = df.resample('D').mean()

    # Resampling to monthly frequency
    df_month = df.resample('M').mean()

    # Resampling to annual frequency
    df_year = df.resample('A-DEC').mean()

    # Resampling to quarterly frequency
    df_Q = df.resample('Q-DEC').mean()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    # PLOTS
    fig = plt.figure(figsize=[15, 7])
    plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)

    plt.subplot(221)
    plt.plot(df.Weighted_Price, '-', label='By Days')
    plt.legend()

    plt.subplot(222)
    plt.plot(df_month.Weighted_Price, '-', label='By Months')
    plt.legend()

    plt.subplot(223)
    plt.plot(df_Q.Weighted_Price, '-', label='By Quarters')
    plt.legend()

    plt.subplot(224)
    plt.plot(df_year.Weighted_Price, '-', label='By Years')
    plt.legend()

    # plt.tight_layout()
    plt.show()

    plt.figure(figsize=[15,7])
    sm.tsa.seasonal_decompose(df_month.Weighted_Price).plot()
    print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Box-Cox Transformations
    df_month['Weighted_Price_box'], lmbda = stats.boxcox(df_month.Weighted_Price)
    print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])

    # Seasonal differentiation
    df_month['prices_box_diff'] = df_month.Weighted_Price_box - df_month.Weighted_Price_box.shift(12)
    print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.prices_box_diff[12:])[1])

    # Regular differentiation
    df_month['prices_box_diff2'] = df_month.prices_box_diff - df_month.prices_box_diff.shift(1)
    plt.figure(figsize=(15,7))

    # STL-decomposition
    sm.tsa.seasonal_decompose(df_month.prices_box_diff2[13:]).plot()   
    print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.prices_box_diff2[13:])[1])

    plt.show()



    # --- PYCARET AUTOML ──────────────────────────────────────

    from pycaret.time_series import *

    ts_setup = setup(data=df, target='Close', fh=12, session_id=42, verbose=False)

    # Compare models and select best
    best_model = compare_models()

    # Display comparison results
    print(best_model)

    # Plot forecast
    plot_model(best_model, plot='forecast')

    # Finalize the model
    final_model = finalize_model(best_model)

    # Make predictions
    predictions = predict_model(final_model)
    print(predictions)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full pipeline for Predicting the price of bitcoin")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
