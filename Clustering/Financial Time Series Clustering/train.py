#!/usr/bin/env python3
"""
Model training for 9 Clustering financial time series

Auto-generated from: 9 Clustering financial time series.ipynb
Project: 9 Clustering financial time series
Category: Clustering | Task: clustering
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import os
import pandas as pd
import numpy as np
import random
import itertools
from arch import arch_model
from scipy.stats import shapiro
from scipy.stats import probplot
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from pycaret.clustering import *

# ======================================================================
# TRAINING PIPELINE
# ======================================================================

def main():
    """Run the training pipeline."""
    USE_AUTOML = True  # Set to False to skip AutoML comparison

    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- ADDITIONAL PROCESSING ───────────────────────────────

    import os
    import pandas as pd
    import numpy as np
    import random
    import itertools
    from arch import arch_model
    from scipy.stats import shapiro
    from scipy.stats import probplot
    from statsmodels.stats.diagnostic import het_arch
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox

    from matplotlib import pyplot as plt
    plt.style.use('fivethirtyeight') 
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10



    # --- DATA LOADING ────────────────────────────────────────

    path = '../../data/clustering_financial_time_series/all_stocks_5yr.csv'
    csvs = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

    df = pd.DataFrame()
    for file in random.sample(range(1, len(csvs)), 8):
        stock_df = load_dataset('clustering_financial_time_series')
        stock_df.index = pd.DatetimeIndex(stock_df.date)
        name = stock_df['Name'].iloc[0]
        df[name] = stock_df['close']

    df.plot(figsize=(10, 5), title='Closing Price for 8 Random Stocks')
    df.head()

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    stock = 'TDG'
    file_path = '../../data/clustering_financial_time_series/all_stocks_5yr.csv'
    file_name = f'{stock}_data.csv'
    file_full_path = os.path.join(file_path, file_name)

    df = pd.read_csv(file_full_path)
    df.index = pd.DatetimeIndex(df.date)
    df = df.drop(columns=['open', 'high', 'low', 'volume', 'date', 'Name'])
    df['pct_change'] = 100 * df['close'].pct_change()
    df.dropna(inplace=True)

    df['close'].plot(figsize=(10, 5), title=f'{stock} Closing Price 2013-2018')
    plt.show()

    df['pct_change'].plot(figsize=(10, 5), title=f'{stock} Percent Change in Closing Price')
    plt.show()

    acf = plot_acf(df['pct_change'], lags=30)
    pacf = plot_pacf(df['pct_change'], lags=30)
    acf.suptitle(f'{stock} Percent Change Autocorrelation and Partial Autocorrelation', fontsize=20)
    acf.set_figheight(5)
    acf.set_figwidth(15)
    pacf.set_figheight(5)
    pacf.set_figwidth(15)
    plt.show()



    # --- AUTOML COMPARISON ────────────────────────────────────

    if USE_AUTOML:

        try:

            # --- PYCARET AUTOML ──────────────────────────────

            from pycaret.clustering import *

            clust_setup = setup(data=df, normalize=True, session_id=42, verbose=False)

            # Create K-Means model
            kmeans_model = create_model('kmeans')
            print(kmeans_model)

            # Assign cluster labels to data
            clustered_df = assign_model(kmeans_model)
            clustered_df.head()

            # Evaluate clustering
            plot_model(kmeans_model, plot='elbow')

            # Silhouette plot
            plot_model(kmeans_model, plot='silhouette')

            # Distribution plot
            plot_model(kmeans_model, plot='distribution')



        except ImportError:

            print('[AutoML] LazyPredict/PyCaret not installed — skipping AutoML block')

        except Exception as _automl_err:

            print(f'[AutoML] AutoML block failed: {_automl_err}')


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model training for 9 Clustering financial time series")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
