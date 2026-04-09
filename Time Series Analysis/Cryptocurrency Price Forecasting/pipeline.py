#!/usr/bin/env python3
"""
Full pipeline for Cryptocurrency Price Forecasting

Auto-generated from: code.ipynb
Project: Cryptocurrency Price Forecasting
Category: Time Series Analysis | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import statsmodels.api as sm
from itertools import product
from math import sqrt
from sklearn.metrics import mean_squared_error
import warnings
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

    # --- EVALUATION ──────────────────────────────────────────

    import pandas as pd
    from pandas import DataFrame
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime, timedelta
    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from scipy import stats
    import statsmodels.api as sm
    from itertools import product
    from math import sqrt
    from sklearn.metrics import mean_squared_error

    import warnings
    warnings.filterwarnings('ignore')


    colors = ["windows blue", "amber", "faded green", "dusty purple"]
    sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12,
                "xtick.labelsize" : 14, "ytick.labelsize" : 14 })



    # --- DATA LOADING ────────────────────────────────────────

    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    df = load_dataset('cryptocurrency_price_forecasting')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df.sample(5)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Extract the bitcoin data only
    btc=df[df['Symbol']=='BTC']
    # Drop some columns
    btc.drop(['Volume', 'Market Cap'],axis=1,inplace=True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Resampling to monthly frequency
    btc_month = btc.resample('M').mean()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #seasonal_decompose(btc_month.close, freq=12).plot()
    seasonal_decompose(btc_month.Close, model='additive').plot()
    print("Dickey–Fuller test: p=%f" % adfuller(btc_month.Close)[1])



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Box-Cox Transformations
    btc_month['close_box'], lmbda = stats.boxcox(btc_month.Close)
    print("Dickey–Fuller test: p=%f" % adfuller(btc_month.close_box)[1])

    # Seasonal differentiation (12 months)
    btc_month['box_diff_seasonal_12'] = btc_month.close_box - btc_month.close_box.shift(12)
    print("Dickey–Fuller test: p=%f" % adfuller(btc_month.box_diff_seasonal_12[12:])[1])

    # Seasonal differentiation (3 months)
    btc_month['box_diff_seasonal_3'] = btc_month.close_box - btc_month.close_box.shift(3)
    print("Dickey–Fuller test: p=%f" % adfuller(btc_month.box_diff_seasonal_3[3:])[1])

    # Regular differentiation
    btc_month['box_diff2'] = btc_month.box_diff_seasonal_12 - btc_month.box_diff_seasonal_12.shift(1)

    # STL-decomposition
    seasonal_decompose(btc_month.box_diff2[13:]).plot()
    print("Dickey–Fuller test: p=%f" % adfuller(btc_month.box_diff2[13:])[1])

    #autocorrelation_plot(btc_month.close)
    plot_acf(btc_month.Close[13:].values.squeeze(), lags=12)
    plt.tight_layout()

    # Initial approximation of parameters using Autocorrelation and Partial Autocorrelation Plots
    ax = plt.subplot(211)
    # Plot the autocorrelation function
    #sm.graphics.tsa.plot_acf(btc_month.box_diff2[13:].values.squeeze(), lags=48, ax=ax)
    plot_acf(btc_month.box_diff2[13:].values.squeeze(), lags=12, ax=ax)
    ax = plt.subplot(212)
    #sm.graphics.tsa.plot_pacf(btc_month.box_diff2[13:].values.squeeze(), lags=48, ax=ax)
    plot_pacf(btc_month.box_diff2[13:].values.squeeze(), lags=12, ax=ax)
    plt.tight_layout()



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
    _parser = _ap.ArgumentParser(description="Full pipeline for Cryptocurrency Price Forecasting")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
