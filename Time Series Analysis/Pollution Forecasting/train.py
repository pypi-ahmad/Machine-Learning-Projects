#!/usr/bin/env python3
"""
Model training for Pollution Forecasting

Auto-generated from: code.ipynb
Project: Pollution Forecasting
Category: Time Series Analysis | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import tensorflow as tf
import numpy as np #Linear Algebra
import matplotlib.pyplot as plt #Data visualization
import pandas as pd #data manipulation
import warnings
from pycaret.time_series import *

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
    tf.random.set_seed(42)

    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Import Libraries
    import tensorflow as tf
    import numpy as np #Linear Algebra
    import matplotlib.pyplot as plt #Data visualization
    import pandas as pd #data manipulation

    import warnings
    warnings.filterwarnings('ignore') #Ignore warnings

    #Make sure Tensorflow is version 2.0 or higher
    print('Tensorflow Version:', tf.__version__)



    # --- DATA LOADING ────────────────────────────────────────

    #Reads in Pollution csv
    pollution = load_dataset('pollution_forecasting')
    #Filters for only pm25 values in Jeongnim-Dong City, sorted by date
    pollution = pollution[pollution.City == 'Jeongnim-Dong'].pm25.sort_index()
    #starts the dataset at 2018 and ends in 2022(due to breaks in data in previous years)
    start = pd.to_datetime('2018-01-01')
    end = pd.to_datetime('2022-01-01')
    pollution = pollution[start:end]
    print('SAMPLE OF TIME SERIES DATA:')
    pollution.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Checks for and imputes missing dates
    a = pd.date_range(start="2018-01-01", end="2022-01-01", freq="D") #continous dates
    b = pollution.index #our time series
    diff_dates = a.difference(b) # finds what in 'a' is not in 'b'

    td = pd.Timedelta(1, "d") #1 day
    for date in diff_dates:
        prev_val = pollution[date-td] #takes the previous value
        pollution[date] = prev_val #imputes previous value

    pollution.sort_index(inplace=True)
    #sets the time index frequency as daily
    pollution.freq = "D"

    #Split the time series data into a train and test set
    end_train_ix = pd.to_datetime('2020-12-31')
    train = pollution[:end_train_ix] # Jan 2018-2021
    test = pollution[end_train_ix:] # Jan 2021-2022



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #Creates a windowed dataset from the time series data
    WINDOW = 14 #the window value... 14 days

    #converts values to TensorSliceDataset
    train_data = tf.data.Dataset.from_tensor_slices(train.values)

    #takes window size + 1 slices of the dataset
    train_data = train_data.window(WINDOW+1, shift=1, drop_remainder=True)

    #flattens windowed data by batching
    train_data = train_data.flat_map(lambda x: x.batch(WINDOW+1))

    #creates features and target tuple
    train_data = train_data.map(lambda x: (x[:-1], x[-1]))

    #shuffles dataset
    train_data = train_data.shuffle(1_000)

    #creates batches of windows
    train_data = train_data.batch(32).prefetch(1)



    # --- PYCARET AUTOML ──────────────────────────────────────

    from pycaret.time_series import *

    ts_setup = setup(data=pollution, target='None', fh=12, session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Pollution Forecasting")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
